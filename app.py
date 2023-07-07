import torch, PIL
import gradio as gr

title = "OctoBERT"
description = """Interactive Demo for OctoBERT. This base model is trained only on Flickr-30k."""
# examples =[
# ['swing.jpg','The woman stands outdoors, next to a child in a <mask>.'],
# ['tennis.jpg', 'A woman in blue shorts and white shirt holds a tennis racket on a blue <mask> court.'],
# ['birthday.jpg', 'The smiling <mask> is celebrating her <mask> party with friends, surrounded by balloons and a <mask> with candles.'],
# ['skate.jpg', 'A person in a rainbow colored snowsuit is snowboarding down a <mask> slope.'],
# ['street.jpg', 'A man with <mask> plays with a little girl while walking down the street, while an Asian woman walks ahead of them.'],
# ['dog.jpg', 'A black dog stands on a <mask>, green fields behind him.'],
# ]
device = "cuda" if torch.cuda.is_available() else "cpu"
model, img_transform, tokenizer, post_processor, plot_results = torch.hub.load('Jiayi-Pan/RefCloze_Pub', 'flickr_base_model')
model = model.to(device)

# url = "https://i.imgur.com/G0rWulu.jpg"
# img = PIL.Image.open(urlopen(url))
# caption = "three<mask>in a room"

def plot_inference(img, caption):
    imgs_tensor = img_transform(img).to(device).unsqueeze(0)
    tokens_tensor = tokenizer(caption, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(imgs_tensor, tokens_tensor, one_pass=True)
    processed_outputs = post_processor(outputs, img, tokenizer)
    vis = plot_results(img, processed_outputs, save_path="numpy_array")
    return vis, processed_outputs['cap']


gr.Interface(
    plot_inference, 
    [gr.inputs.Image(type="pil", label="Input"), gr.inputs.Textbox(label="input text")], 
    [gr.outputs.Image(type="numpy", label="Output"), gr.outputs.Textbox(label="Predicted Words")],
    title=title,
    description=description,
    # examples=examples,
    cache_examples=False,
    ).launch()