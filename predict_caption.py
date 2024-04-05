from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from openai import OpenAI

model = VisionEncoderDecoderModel.from_pretrained('vit-gpt2-image-captioning')
feature_extractor = ViTImageProcessor.from_pretrained(
    'vit-gpt2-image-captioning')
tokenizer = AutoTokenizer.from_pretrained('vit-gpt2-image-captioning')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {'max_length': max_length, 'num_beams': num_beams}

def predict_step(image,purpose):

    pixel_values = feature_extractor(
        images=[image], return_tensors='pt').pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]

    print(preds[0])

    client = OpenAI()

    prompt="Generate 2 quotes for an image which contains this context : "+preds[0]+" and the purpose     of the social media post is : "+purpose+" also give top hashtags"
    response=client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a helpful social media post caption provider."},
                {"role": "user", "content": prompt}
            ]
        )

    # Extract the generated text from the response
    print(response)
    generated_text = response.choices[0].message
    texts = [item.split('"')[1] for item in generated_text.content.split('\n') if item.strip()]
    op=''

    for t in texts:
        op=op+t+"\n"
    return op
