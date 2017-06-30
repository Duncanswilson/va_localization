# va_localization
using latent representations of images to have systems learn where the hell they =

## Localization Via Latent Representation
 
### Work Outline
- Make sure dataset is all standardized and nice. **Richard**
- Re-upload dataset to Dropbox. **Richard**
- Push the code to handle the dataset in the Tensorflow standard. **Richard**  
- Make sure VA is set up in modular fashion and have it work on the data and can consistently generate interesting outputs. **Duncan** (eta: 1-2 days) 
	- Richard can you send me that o.o.-tf blog post?    
	- Q: Should this mean that once the VA is set up & converging with low loss that it will be outputting something that could be visualized as a rough approximation of a dashcam image? If so, that already seems pretty interesting to me, but probably not novel. 
	- A thought if this fails: we may be able to DeepDream on Pix2Pix or another GAN for the latent rep if VAs have issues. honestly comparing the VA method to that, depending on how we're looking for time against the deadline, seems fun. 
- Write up a simple Image -> ConvNet -> GPS model to see if we don't need the latent representation. **Duncan preferred** (eta: 3 days) 
- Set up LSTM or other appropriate model to map from (Image, Latent_Rep) -> GPS **whoever gets to this first**
- See how the above stuff performs and plan again from here. (eta: 5/6 days)

### Repo Details
- A small dataset consisting of dashcam images and GPS coordinates recorded on the morning of January 13'th and stored in ```/data```. 
- A very rough Frankenstein-style, copypasta'd Variational AutoEncoder Jupyter notebook can be found in ```/models```, soon to be corrected. 
	- also soon to include conv net & lstm model too
- All other supporting files can be found in ```/utils```.


