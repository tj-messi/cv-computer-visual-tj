
PitVQA Dataset

Our PitVQA dataset comprises 25 videos of endoscopic pituitary surgeries from the The National Hospital of Neurology and Neurosurgery in London, United Kingdom. All videos were annotated for the surgical phases, steps, instruments present and operation notes guided by a standardised annotation framework, which was derived from a preceding international consensus study on pituitary surgery workflow. Annotation was performed collaboratively by 2 neurosurgical residents with operative pituitary experience and checked by an attending neurosurgeon. We extracted image frames from each video at 1 fps and removed any frames that were blurred or occluded. Ultimately, we obtained a total of 109,173 frames, with the videos of minimum and maximum length yielding 2,443 and 7,179 frames, respectively. We acquired frame-wise question-answer pairs for all the categories of the annotation. Overall, there are 884,242 question-answer pairs from 109,173 frames, which is around 8 pairs for each frame. There are 59 classes overall, including 4 phases, 15 steps, 18 instruments, 3 variations of instruments present in a frame, 5 positions of the instruments, and 14 operation notes in the annotation classes.


Acknowledgement
PitVQA images are derived from MICCAI PitVis challenge:
PitVis Paper: https://arxiv.org/abs/2409.01184
PitVis Challenge: https://www.synapse.org/Synapse:syn51232283
PitVis Dataset: https://doi.org/10.5522/04/26531686


If you use this code for your research, please cite our paper.

He, Runlong, Mengya Xu, Adrito Das, Danyal Z. Khan, Sophia Bano, Hani J. Marcus, Danail Stoyanov, Matthew J. Clarkson, and Mobarakol Islam. "PitVQA: Image-grounded Text Embedding LLM for Visual Question Answering in Pituitary Surgery." MICCAI 2024.
