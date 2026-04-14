from typing import Dict, Any, Union
from dataclasses import dataclass
from tqdm import tqdm
import torch, peft
import logging, copy, time

from adv_trainers.data import DatasetWrapper, DatacollatorConfig, build_datacollator, Loader
from adv_trainers.utils import build_optimizer, build_scheduler


class ContinuousAttacker:
    def __init__(
        self,
        steps: int,
        step_size: float,
        radius: float,
        norm_type: str = "l-infty",
        disable_tqdm: bool = False,
    ):
        self.steps = steps
        self.step_size = step_size
        self.radius = radius
        self.norm_type = norm_type
        self.disable_tqdm = disable_tqdm

        assert norm_type in ["l-infty", "l2"]

    def attack_embeds(
        self,
        model,
        tokenizer,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
        device: Union[str, torch.device],
    ):
        reqs_grad = []
        for pp in model.parameters():
            reqs_grad.append(pp.requires_grad)
            pp.requires_grad = False
        model.eval()

        # prepare parameters
        vocab_size = model.vocab_size
        embedding_layer = model.get_input_embeddings()

        prompt_embeds = embedding_layer(prompt_ids)
        B, pro_len, embed_dim = prompt_embeds.shape
        dtype = prompt_embeds.dtype

        target_embeds = embedding_layer(target_ids)
        tar_len = target_embeds.shape[1]

        input_mask = torch.cat([prompt_mask, torch.ones_like(target_mask, dtype=torch.int64)], dim=1)

        # if self.norm_type == "l-infty":
        #     adv_prompt_embeds = (torch.rand_like(prompt_embeds) * 2 - 1) * self.radius
        # elif self.norm_type == "l2":
        #     adv_prompt_embeds = (torch.rand_like(prompt_embeds) * 2 - 1) * (self.radius / (self.steps + 1e-6))
        adv_prompt_embeds = torch.zeros_like(prompt_embeds)

        adv_prompt_embeds = prompt_embeds + adv_prompt_embeds
        adv_prompt_embeds.requires_grad_()
        self._clip_embeds_(adv_prompt_embeds, prompt_embeds, prompt_mask)

        pbar = tqdm(range(self.steps), disable=self.disable_tqdm)
        for step in pbar:
            input_embeds = torch.cat([adv_prompt_embeds, target_embeds], dim=1)
            out_logits = model(
                inputs_embeds=torch.cat([adv_prompt_embeds, target_embeds], dim=1),
                attention_mask=input_mask,
            )["logits"]

            out_logits = out_logits[:, -(tar_len+1) : -1]
            out_logps = out_logits.log_softmax(dim=-1)
            out_logps = torch.gather(out_logps, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
            loss = -(out_logps * target_mask).mean()
            gd = torch.autograd.grad(loss, [adv_prompt_embeds])[0]

            # gd = gd / (gd.view(B, -1).abs().max(dim=-1).values + 1e-6).reshape(B, 1, 1)
            self._clip_gd_(gd)
            # gd = gd.sign()
            adv_prompt_embeds.data.add_(gd, alpha= -self.step_size)
            self._clip_embeds_(adv_prompt_embeds, prompt_embeds, prompt_mask)

            # # ==== debug begin ====
            # print(f"loss = {loss.item()}", flush=True)
            # # ===== debug end =====

        # re-open model autograd
        for pp, rq_gd in zip(model.parameters(), reqs_grad):
            pp.requires_grad = rq_gd

        if self.steps == 0 or self.radius == 0:
            return torch.zeros_like(adv_prompt_embeds.data)

        return adv_prompt_embeds.data - prompt_embeds

    @torch.no_grad()
    def _clip_embeds_(self, adv_prompt_embeds, prompt_embeds, prompt_mask):
        # shape: [B, L, dim]
        adv_prompt_embeds -= prompt_embeds
        adv_prompt_embeds *= prompt_mask.unsqueeze(-1)
        if self.norm_type == "l-infty":
            adv_prompt_embeds.clamp_(-self.radius, self.radius)
        elif self.norm_type == "l2":
            norm = (adv_prompt_embeds ** 2).sum(dim=-1).sqrt().unsqueeze(-1) # shape: [B, L, 1]
            mask = (norm >= self.radius)
            adv_prompt_embeds.data = (mask^True) * adv_prompt_embeds + mask * adv_prompt_embeds / (norm+1e-8) * self.radius
        adv_prompt_embeds += prompt_embeds

    @torch.no_grad()
    def _clip_gd_(self, gd):
        # shape: [B, L, dim]
        if self.norm_type == "l-infty":
            gd.sign_()
        elif self.norm_type == "l2":
            norm = (gd ** 2).sum(dim=-1).sqrt().unsqueeze(-1) # shape: [B, L, 1]
            gd.data /= (norm + 1e-6)
            # gd.sign_()


@dataclass
class CATTrainerConfig:
    train_steps: int
    report_freq: int
    batch_size: int
    optimizer_config: Dict
    scheduler_config: Dict
    datacollator_config: DatacollatorConfig
    warm_embed_steps: int = 0
    attacker_config: Dict = None
    utility_batch_size: int = None
    utility_split_bs: int = None
    utility_max_length: int = None
    alpha_toward: float = 1.0
    alpha_away: float = 1.0
    alpha_utility: float = 1.0
    alpha_embed: float = 0.0
    cutoff_away: float = 5.0
    cutoff_toward: float = 0.5

    def __post_init__(self):
        self.datacollator_config = DatacollatorConfig(**self.datacollator_config)

        if isinstance(self.alpha_toward, str):
            self.alpha_toward = float(self.alpha_toward)

        if isinstance(self.alpha_away, str):
            self.alpha_away = float(self.alpha_away)

        if isinstance(self.alpha_utility, str):
            self.alpha_utility = float(self.alpha_utility)

        if isinstance(self.alpha_embed, str):
            self.alpha_embed = float(self.alpha_embed)

        if isinstance(self.cutoff_away, str):
            self.cutoff_away = float(self.cutoff_away)


class CATTrainer:
    def __init__(
        self,
        trainer_config: CATTrainerConfig,
        model,
        tokenizer,
        device: Union[str, torch.device],
        trainset: DatasetWrapper,
        utilityset: DatasetWrapper = None,
        logger: logging.Logger = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.trainset = trainset
        self.logger = logger
        self.logs = []

        trainer_config = copy.deepcopy(trainer_config)

        self.train_steps = trainer_config.train_steps
        self.warm_embed_steps = trainer_config.warm_embed_steps
        self.current_step = 0

        self.report_freq = trainer_config.report_freq

        self.optimizer_config = trainer_config.optimizer_config
        self.optimizer = build_optimizer(
            parameters=model.parameters(), **self.optimizer_config)

        self.scheduler_config = trainer_config.scheduler_config
        self.scheduler = build_scheduler(
            optimizer=self.optimizer, **self.scheduler_config)

        self.attacker_config = trainer_config.attacker_config
        self.attacker = None
        if self.attacker_config is not None:
            self.attacker = ContinuousAttacker(**self.attacker_config)

        self.trainset = trainset
        self.train_datacollator = build_datacollator(
            trainer_config.datacollator_config,
            tokenizer=self.tokenizer,
            is_adv=True,
        )
        self.trainloader = Loader(
            dataset=trainset,
            batch_size=trainer_config.batch_size,
            train=True,
            collate_fn=self.train_datacollator,
            num_workers=0,
        )

        self.utilityset = None

        if utilityset is not None:
            self.utilityset = utilityset
            self.utility_datacollator = build_datacollator(
                trainer_config.datacollator_config,
                tokenizer=self.tokenizer,
                is_adv=False,
            )
            self.utilityloader = Loader(
                dataset=utilityset,
                batch_size=trainer_config.utility_batch_size,
                train=True,
                collate_fn=self.utility_datacollator,
                num_workers=0,
            )
            self.utility_split_bs = trainer_config.utility_split_bs
            self.utility_max_length = trainer_config.utility_max_length

            self.alpha_toward = trainer_config.alpha_toward
            self.alpha_away = trainer_config.alpha_away
            self.alpha_utility = trainer_config.alpha_utility
            self.alpha_embed = trainer_config.alpha_embed

            self.cutoff_toward = trainer_config.cutoff_toward
            self.cutoff_away = trainer_config.cutoff_away

    def train(self, next_steps: int = -1):
        all_steps = self.train_steps + self.warm_embed_steps
        if next_steps < 0:
            next_steps = all_steps
        next_steps = min(next_steps, all_steps - self.current_step)

        cumu_time = 0

        for i in range(next_steps):
            log = {}

            self.current_step += 1
            log["step"] = self.current_step

            # time counting begin
            torch.cuda.synchronize()
            start_time = time.time()

            inputs = next(self.trainloader)
            inputs = {k : v.to(self.device) for k, v in inputs.items()}

            losses = {}

            self.optimizer.zero_grad()

            # if self.current_step <= self.warm_embed_steps:
            if True:
                tmp_losses = self._grad_embed_loss()
                for k, v in tmp_losses.items(): losses[k] = v

            # if self.current_step > self.warm_embed_steps:
            if True:
                adv_prompt_perturb = None
                if self.attacker is not None:
                    adv_prompt_perturb = self.attacker.attack_embeds(
                        model       = self.model,
                        tokenizer   = self.tokenizer,
                        prompt_ids  = inputs["prompt_ids"],
                        prompt_mask = inputs["prompt_mask"],
                        target_ids  = inputs["harmful_ids"],
                        target_mask = inputs["harmful_mask"],
                        device      = self.device,
                    )

                adv_inputs = {k:v for k, v in inputs.items()}
                adv_inputs["adv_prompt_perturb"] = adv_prompt_perturb
                tmp_losses = self._grad_adv_loss(adv_inputs)
                for k, v in tmp_losses.items(): losses[k] = v

            utility_inputs = None
            if self.utilityset is not None:
                utility_inputs = next(self.utilityloader)
                utility_inputs = {k : v.to(self.device) for k, v in utility_inputs.items()}
                tmp_losses = self._grad_utility_loss(utility_inputs)
                for k, v in tmp_losses.items(): losses[k] = v

            self.optimizer.step()
            self.scheduler.step()

            # time counting end
            torch.cuda.synchronize()
            end_time = time.time()

            used_time = end_time - start_time
            cumu_time += used_time
            log["used_time_s"] = used_time
            log["cumu_time_s"] = cumu_time

            log["losses"] = losses

            with torch.no_grad():
                embedding_layer = self.model.get_input_embeddings()
                embedding_tokens = embedding_layer.weight.clone()
                if isinstance(embedding_layer, peft.tuners.lora.layer.Embedding):
                    embedding_tokens = embedding_tokens + embedding_layer.get_delta_weight(self.model.active_adapter)
                S = torch.linalg.svdvals(embedding_tokens.data.to(torch.float32))
                log["embed_singular_max"]  = S.max().item()
                log["embed_singular_min"]  = S.min().item()
                log["embed_singular_sum"]  = S.sum().item()
                log["embed_singular_mean"] = S.mean().item()
                log["embed_singular_std"]  = S.std().item()

            self.logs.append(log)
            if (self.logger is not None) and ((self.current_step) % self.report_freq == 0):
                self.logger.info(f"Step: [{self.current_step}/{all_steps}]")
                for k, v in log.items():
                    self.logger.info(f"{k}: {v}")

    def _grad_adv_loss(self, adv_inputs: dict) -> dict:
        self.model.train()

        prompt_embeds = self.model.get_input_embeddings()(adv_inputs["prompt_ids"])
        if adv_inputs["adv_prompt_perturb"] is not None:
            print("haha: adv_prompt_perturb_max = {}, min = {}"
                  .format(adv_inputs["adv_prompt_perturb"].max(), adv_inputs["adv_prompt_perturb"].min()))
            prompt_embeds = prompt_embeds + adv_inputs["adv_prompt_perturb"]

        input_embeds = torch.cat([ prompt_embeds, self.model.get_input_embeddings()(adv_inputs["benign_ids"]), ], dim=1)
        input_mask = torch.cat([ adv_inputs["prompt_mask"], torch.ones_like(adv_inputs["benign_mask"]), ], dim=1)

        logits = self.model(inputs_embeds=input_embeds, attention_mask=input_mask).logits
        logits = logits[:, -(adv_inputs["benign_ids"].shape[-1] + 1) : -1]
        logps = torch.gather(logits.log_softmax(dim=-1), dim=-1, index=adv_inputs["benign_ids"].unsqueeze(-1)).squeeze(-1)
        toward_loss = -(logps * adv_inputs["benign_mask"]).sum() / adv_inputs["benign_mask"].sum()
        # toward_loss = (toward_loss>self.cutoff_toward) * (0.9*self.cutoff_toward + 0.1*toward_loss) + (toward_loss<=self.cutoff_toward) * toward_loss
        # toward_loss = (toward_loss<self.cutoff_toward) * (0.99*self.cutoff_toward + 0.01*toward_loss) + (toward_loss>=self.cutoff_toward) * toward_loss
        toward_loss = (toward_loss<self.cutoff_toward) * (0.999*self.cutoff_toward + 0.001*toward_loss) + (toward_loss>=self.cutoff_toward) * toward_loss

        alpha_toward_loss = toward_loss * self.alpha_toward
        alpha_toward_loss.backward()
        toward_loss = toward_loss.item()

        prompt_embeds = self.model.get_input_embeddings()(adv_inputs["prompt_ids"])
        if adv_inputs["adv_prompt_perturb"] is not None:
            prompt_embeds = prompt_embeds + adv_inputs["adv_prompt_perturb"]

        input_embeds = torch.cat([ prompt_embeds, self.model.get_input_embeddings()(adv_inputs["harmful_ids"]), ], dim=1)
        input_mask = torch.cat([ adv_inputs["prompt_mask"], torch.ones_like(adv_inputs["harmful_mask"]), ], dim=1)

        logits = self.model(inputs_embeds=input_embeds, attention_mask=input_mask).logits
        logits = logits[:, -(adv_inputs["harmful_ids"].shape[-1] + 1) : -1]

        logps = torch.gather(logits.log_softmax(dim=-1), dim=-1, index=adv_inputs["harmful_ids"].unsqueeze(-1)).squeeze(-1)

        away_loss = -(logps * adv_inputs["harmful_mask"]).sum() / adv_inputs["harmful_mask"].sum()

        # force to cutoff away_loss
        # away_loss = (away_loss>self.cutoff_away) * (0.9*self.cutoff_away + 0.1*away_loss) + (away_loss<=self.cutoff_away) * away_loss
        # away_loss = (away_loss>self.cutoff_away) * (0.99*self.cutoff_away + 0.01*away_loss) + (away_loss<=self.cutoff_away) * away_loss
        away_loss = (away_loss>self.cutoff_away) * (0.999*self.cutoff_away + 0.001*away_loss) + (away_loss<=self.cutoff_away) * away_loss

        alpha_away_loss = away_loss * (-self.alpha_away)
        # alpha_away_loss = away_loss * self.alpha_away
        alpha_away_loss.backward()
        away_loss = away_loss.item()

        return {
            "toward_loss"  : toward_loss,
            "away_loss"    : away_loss,
        }

    def _grad_utility_loss(self, utility_inputs: dict = None) -> dict:
        self.model.train()

        utility_loss = 0
        if utility_inputs is not None:
            u_bs = len(utility_inputs["input_ids"])
            u_cnt = utility_inputs["input_mask"].sum()
            sp_bs = self.utility_split_bs
            if sp_bs is None:
                sp_bs = u_bs

            print("u_ids.shape = ", utility_inputs["input_ids"].shape)

            utility_loss = 0
            for be in range(0, u_bs, sp_bs):
                ed = min(u_bs, be + sp_bs)

                u_ids = utility_inputs["input_ids"][be:ed]
                u_mask = utility_inputs["input_mask"][be:ed]

                if self.utility_max_length is not None:
                    max_L = self.utility_max_length
                    if u_ids.shape[1] > max_L:
                        u_ids = u_ids[:, : max_L]
                        u_mask = u_mask[:, : max_L]

                u_logits = self.model(input_ids=u_ids).logits
                u_logps = u_logits.log_softmax(dim=-1)
                u_logps = torch.gather(u_logps[:, :-1], dim=-1, index=u_ids[:, 1:].unsqueeze(-1)).squeeze(-1) # shape: [ed-be, L-1]
                u_mask = u_mask[:, 1:]

                u_loss = -(u_logps * u_mask).sum() / u_cnt
                u_alpha_loss = u_loss * self.alpha_utility
                u_alpha_loss.backward()
                utility_loss += u_loss.item()

        return { "utility_loss" : utility_loss }

    def _grad_embed_loss(self) -> dict:
        if self.alpha_embed <= 0:
            return {}

        self.model.train()

        embed_loss = 0.0
        embedding_layer = self.model.get_input_embeddings()
        embed_pms = []
        embed_pms_grads = []
        # if self.embed_alpha > 0 and isinstance(embedding_layer, peft.tuners.lora.layer.Embedding):
        if isinstance(embedding_layer, peft.tuners.lora.layer.Embedding):
            embed_pms = [pp for pp in embedding_layer.parameters() if pp.requires_grad is True]
            embedding_tokens = embedding_layer.weight.clone()
            embedding_tokens = embedding_tokens + embedding_layer.get_delta_weight(self.model.active_adapter)

            S = torch.linalg.svdvals(embedding_tokens)
            embed_loss = S.std()

            alpha_embed_loss = embed_loss * self.alpha_embed
            alpha_embed_loss.backward()
            embed_loss = embed_loss.item()

        return { "embed_loss" : embed_loss }

    def state_dict(self) -> Dict[str, Any]:
        trainer_state_dict = {
            "current_step" : self.current_step,
            "optimizer"    : self.optimizer.state_dict(),
            "scheduler"    : self.scheduler.state_dict(),
        }
        return self.model.state_dict(), trainer_state_dict

    def load_state_dict(
        self,
        model_state_dict: Dict[str, torch.Tensor] = None,
        trainer_state_dict: Dict[str, Any] = None,
    ):
        if model_state_dict is not None:
            self.model.load_state_dict(model_state_dict)

        if trainer_state_dict is not None:
            self.current_step = trainer_state_dict["current_step"]
            self.optimizer.load_state_dict(trainer_state_dict["optimizer"])
            self.scheduler.load_state_dict(trainer_state_dict["scheduler"])
