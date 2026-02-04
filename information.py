'''
mlp-Gan is the generative adversarial network in which we uise Generator and descriminator 
now generator judges the fake Data while decriminator check fake vs real data 
both are use similatnaously to work together and helps predict the output 
as we have defined inside the program(generator, descriminator )

MLP - multi layer perception This model have linear layer , has flatten data 

Channel- channel is define as color like if the color is blacl and white we take (1,28,28) and if the color is in rgb we take(3,28,28) w
while the other componet are weight and height 
so transform.greyscale() is use for that to teach the model 

INTERVIEW EXPLANATION
“I built a Generative Adversarial Network using PyTorch to generate handwritten digit images. 
The project consists of a Generator that learns to create fake images from random noise and
 a Discriminator that learns to distinguish real images from generated ones. 
Both networks are trained together in an adversarial setup on the MNIST dataset using GPU acceleration.
 I focused on understanding the full training pipeline, data loading, model design, and GAN stability.”

 COMMON FOLLOW-UP QUESTIONS (WITH PERFECT ANSWERS)
❓ Why did you choose GANs?

“GANs are a foundational generative model and help understand how adversarial learning works.
 I wanted hands-on experience with training instability, loss behavior, and generator–discriminator balance.”

❓ Why MLP GAN and not CNN?

“I intentionally started with an MLP-based GAN to understand the core GAN mechanics.
 CNN-based GANs like DCGAN are more powerful, but the logic is easier to grasp when starting with fully connected layers.”

This answer is 🔥

❓ What challenges did you face?

“The main challenges were training stability and performance.
 I had to ensure tensors and models were on the same device, manage GPU usage correctly, 
 and understand why small models don’t fully utilize the GPU. Debugging device mismatches taught me a lot about PyTorch internals.”

That shows real debugging experience.

❓ How did you evaluate the model?

“GANs don’t have a direct accuracy metric, so I evaluated the model visually by i
nspecting generated images over epochs and checking whether they became more structured and diverse.”

Correct and professional.

❓ How would you improve this project?

“I would upgrade it to a convolutional GAN like DCGAN for better spatial feature learning,
 experiment with different loss functions, and possibly extend it to conditional GANs.”
'''


