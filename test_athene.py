from transformers import LlamaModel, LlamaPreTrainedModel, TextClassificationPipeline
from torch import nn
import torch
from typing import Dict

class AtheneForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.v_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.CLS_ID = 128003
        # Initialize weights and apply final processing
        self.post_init()

    def get_device(self):
        return self.model.device

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
    ):
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        hidden_states = transformer_outputs.hidden_states[-1]
        scores = []
        rewards = self.v_head(hidden_states).squeeze(-1)

        bs = int(input_ids.shape[0])

        for i in range(bs):
            c_inds = (input_ids[i] == self.CLS_ID).nonzero()
            c_ind = c_inds[-1].item()
            scores.append(rewards[i, c_ind])
        scores = torch.stack(scores)
        return {"scores": scores}

# Make a pipeline to handle pre and post-processing
class AtheneRewardPipeline(TextClassificationPipeline):

    def preprocess(self, inputs, **tokenizer_kwargs) -> Dict[str, torch.Tensor]:
        return_tensors = self.framework

        formatted = self.tokenizer.apply_chat_template(inputs, tokenize=False)

        formatted = formatted + self.tokenizer.cls_token

        return self.tokenizer(
            formatted,
            return_tensors=return_tensors,
            max_length=4096,
            padding="longest",
            truncation=True,
        )

    def postprocess(self, model_outputs, function_to_apply=None, top_k=1, _legacy=True):
        return model_outputs["scores"].cpu().float().item()

# Initialize the model
model = AtheneForSequenceClassification.from_pretrained("Nexusflow/Athene-RM-8B", torch_dtype=bfloat16, cache_dir="/hai/scratch/fangwu97/xu/cache")
tokenizer = AutoTokenizer.from_pretrained("Nexusflow/Athene-RM-8B", cache_dir="/hai/scratch/fangwu97/xu/cache")

# Initialize the pipeline
pipe = pipeline(
            task="text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            pipeline_class=AtheneRewardPipeline,
            device_map="auto",
        )

messages = [
    {
        "role": 'user',
        "content": "What is an Athene Noctura? Explain one sentence."
    },
    {
        "role": "assistant",
        "content": "The Athene noctua, also known as the little owl, is a small, nocturnal owl species native to Europe, Asia, and North Africa, characterized by its distinctive facial disk and piercing yellow eyes."
    }
]

print(pipe([messages])) # Print the reward!
