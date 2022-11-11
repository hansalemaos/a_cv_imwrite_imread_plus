# Less trouble reading/writing images with OpenCV

```python
$pip install a-cv-imwrite-imread-plus
from a_cv_imwrite_imread_plus import add_imwrite_plus_imread_plus_to_cv2
add_imwrite_plus_imread_plus_to_cv2()
cv2.imwrite_plus("f:\\ö\\ö\\ö\\öädssdzß.jpg", base64img2cv)

#or:
from a_cv_imwrite_imread_plus import save_cv_image
save_cv_image("f:\\ö\\ö\\ö\\öädssdzß.jpg", base64img2cv)


	Parameters:
		filepath:str
			folders will be created if they don't exist
		image:np.ndarray
			image as np
	Returns:
		filepath:str
```


```python
from a_cv_imwrite_imread_plus import add_imwrite_plus_imread_plus_to_cv2
add_imwrite_plus_imread_plus_to_cv2()
import base64
from PIL import Image
import cv2

#Base64
base64img = r"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAApgAAAKYB3X3/OAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAANCSURBVEiJtZZPbBtFFMZ/M7ubXdtdb1xSFyeilBapySVU8h8OoFaooFSqiihIVIpQBKci6KEg9Q6H9kovIHoCIVQJJCKE1ENFjnAgcaSGC6rEnxBwA04Tx43t2FnvDAfjkNibxgHxnWb2e/u992bee7tCa00YFsffekFY+nUzFtjW0LrvjRXrCDIAaPLlW0nHL0SsZtVoaF98mLrx3pdhOqLtYPHChahZcYYO7KvPFxvRl5XPp1sN3adWiD1ZAqD6XYK1b/dvE5IWryTt2udLFedwc1+9kLp+vbbpoDh+6TklxBeAi9TL0taeWpdmZzQDry0AcO+jQ12RyohqqoYoo8RDwJrU+qXkjWtfi8Xxt58BdQuwQs9qC/afLwCw8tnQbqYAPsgxE1S6F3EAIXux2oQFKm0ihMsOF71dHYx+f3NND68ghCu1YIoePPQN1pGRABkJ6Bus96CutRZMydTl+TvuiRW1m3n0eDl0vRPcEysqdXn+jsQPsrHMquGeXEaY4Yk4wxWcY5V/9scqOMOVUFthatyTy8QyqwZ+kDURKoMWxNKr2EeqVKcTNOajqKoBgOE28U4tdQl5p5bwCw7BWquaZSzAPlwjlithJtp3pTImSqQRrb2Z8PHGigD4RZuNX6JYj6wj7O4TFLbCO/Mn/m8R+h6rYSUb3ekokRY6f/YukArN979jcW+V/S8g0eT/N3VN3kTqWbQ428m9/8k0P/1aIhF36PccEl6EhOcAUCrXKZXXWS3XKd2vc/TRBG9O5ELC17MmWubD2nKhUKZa26Ba2+D3P+4/MNCFwg59oWVeYhkzgN/JDR8deKBoD7Y+ljEjGZ0sosXVTvbc6RHirr2reNy1OXd6pJsQ+gqjk8VWFYmHrwBzW/n+uMPFiRwHB2I7ih8ciHFxIkd/3Omk5tCDV1t+2nNu5sxxpDFNx+huNhVT3/zMDz8usXC3ddaHBj1GHj/As08fwTS7Kt1HBTmyN29vdwAw+/wbwLVOJ3uAD1wi/dUH7Qei66PfyuRj4Ik9is+hglfbkbfR3cnZm7chlUWLdwmprtCohX4HUtlOcQjLYCu+fzGJH2QRKvP3UNz8bWk1qMxjGTOMThZ3kvgLI5AzFfo379UAAAAASUVORK5CYII="
base64img2 = r"iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAApgAAAKYB3X3/OAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAANCSURBVEiJtZZPbBtFFMZ/M7ubXdtdb1xSFyeilBapySVU8h8OoFaooFSqiihIVIpQBKci6KEg9Q6H9kovIHoCIVQJJCKE1ENFjnAgcaSGC6rEnxBwA04Tx43t2FnvDAfjkNibxgHxnWb2e/u992bee7tCa00YFsffekFY+nUzFtjW0LrvjRXrCDIAaPLlW0nHL0SsZtVoaF98mLrx3pdhOqLtYPHChahZcYYO7KvPFxvRl5XPp1sN3adWiD1ZAqD6XYK1b/dvE5IWryTt2udLFedwc1+9kLp+vbbpoDh+6TklxBeAi9TL0taeWpdmZzQDry0AcO+jQ12RyohqqoYoo8RDwJrU+qXkjWtfi8Xxt58BdQuwQs9qC/afLwCw8tnQbqYAPsgxE1S6F3EAIXux2oQFKm0ihMsOF71dHYx+f3NND68ghCu1YIoePPQN1pGRABkJ6Bus96CutRZMydTl+TvuiRW1m3n0eDl0vRPcEysqdXn+jsQPsrHMquGeXEaY4Yk4wxWcY5V/9scqOMOVUFthatyTy8QyqwZ+kDURKoMWxNKr2EeqVKcTNOajqKoBgOE28U4tdQl5p5bwCw7BWquaZSzAPlwjlithJtp3pTImSqQRrb2Z8PHGigD4RZuNX6JYj6wj7O4TFLbCO/Mn/m8R+h6rYSUb3ekokRY6f/YukArN979jcW+V/S8g0eT/N3VN3kTqWbQ428m9/8k0P/1aIhF36PccEl6EhOcAUCrXKZXXWS3XKd2vc/TRBG9O5ELC17MmWubD2nKhUKZa26Ba2+D3P+4/MNCFwg59oWVeYhkzgN/JDR8deKBoD7Y+ljEjGZ0sosXVTvbc6RHirr2reNy1OXd6pJsQ+gqjk8VWFYmHrwBzW/n+uMPFiRwHB2I7ih8ciHFxIkd/3Omk5tCDV1t+2nNu5sxxpDFNx+huNhVT3/zMDz8usXC3ddaHBj1GHj/As08fwTS7Kt1HBTmyN29vdwAw+/wbwLVOJ3uAD1wi/dUH7Qei66PfyuRj4Ik9is+hglfbkbfR3cnZm7chlUWLdwmprtCohX4HUtlOcQjLYCu+fzGJH2QRKvP3UNz8bWk1qMxjGTOMThZ3kvgLI5AzFfo379UAAAAASUVORK5CYII="
base64imgcv = cv2.imread_plus(base64img)
base64img2cv = cv2.imread_plus(base64img2)
base64imgcv = cv2.imread_plus(base64img, channels_in_output=4)
base64img2cv = cv2.imread_plus(base64img2, channels_in_output=4)
base64imgcv = cv2.imread_plus(base64img, channels_in_output=2)
base64img2cv = cv2.imread_plus(base64img2, channels_in_output=2)

#urls
pininterestlogo = "https://camo.githubusercontent.com/7f81f312b05694ccc8cd29e3c3466945ff8e73a13320d3fd0f90c6915bbb4ffb/68747470733a2f2f63646e2e6a7364656c6976722e6e65742f67682f646d68656e647269636b732f7369676e61747572652d736f6369616c2d69636f6e732f69636f6e732f726f756e642d666c61742d66696c6c65642f353070782f70696e7465726573742e706e67"
linkcv1 = cv2.imread_plus(pininterestlogo)
linkcv2 = cv2.imread_plus(pininterestlogo, channels_in_output=4)
linkcv3 = cv2.imread_plus(pininterestlogo, channels_in_output=2)
linkcv4 = cv2.imread_plus(pininterestlogo, channels_in_output=3)

#bytes/raw data
byteimage = base64.b64decode(base64img2)
byteimage1 = cv2.imread_plus(byteimage)
byteimage2 = cv2.imread_plus(byteimage, channels_in_output=4)
byteimage3 = cv2.imread_plus(byteimage, channels_in_output=2)
byteimage4 = cv2.imread_plus(byteimage, channels_in_output=3)

#PIL
pilimage = Image.fromarray(byteimage2)
pilimage1 = cv2.imread_plus(pilimage)
pilimage2 = cv2.imread_plus(pilimage, channels_in_output=4)
pilimage3 = cv2.imread_plus(pilimage, channels_in_output=2)
pilimage4 = cv2.imread_plus(pilimage, channels_in_output=3)

#float images to np.uint8
floatimage = pilimage4.astype(float)
floatimage1 = cv2.imread_plus(floatimage)
floatimage2 = cv2.imread_plus(floatimage, channels_in_output=4)
floatimage3 = cv2.imread_plus(floatimage, channels_in_output=2)
floatimage4 = cv2.imread_plus(floatimage, channels_in_output=3)

#filepath
filepath = "c:\\testestestes.png"
pilimage.save(filepath)
filepath1 = cv2.imread_plus(filepath, bgr_to_rgb=True)
filepath2 = cv2.imread_plus(filepath, channels_in_output=4, bgr_to_rgb=True)
filepath3 = cv2.imread_plus(filepath, channels_in_output=2, bgr_to_rgb=True)
filepath4 = cv2.imread_plus(filepath, channels_in_output=3, bgr_to_rgb=True)



#or:
from a_cv_imwrite_imread_plus import open_image_in_cv
#Base64
base64img = r"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAApgAAAKYB3X3/OAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAANCSURBVEiJtZZPbBtFFMZ/M7ubXdtdb1xSFyeilBapySVU8h8OoFaooFSqiihIVIpQBKci6KEg9Q6H9kovIHoCIVQJJCKE1ENFjnAgcaSGC6rEnxBwA04Tx43t2FnvDAfjkNibxgHxnWb2e/u992bee7tCa00YFsffekFY+nUzFtjW0LrvjRXrCDIAaPLlW0nHL0SsZtVoaF98mLrx3pdhOqLtYPHChahZcYYO7KvPFxvRl5XPp1sN3adWiD1ZAqD6XYK1b/dvE5IWryTt2udLFedwc1+9kLp+vbbpoDh+6TklxBeAi9TL0taeWpdmZzQDry0AcO+jQ12RyohqqoYoo8RDwJrU+qXkjWtfi8Xxt58BdQuwQs9qC/afLwCw8tnQbqYAPsgxE1S6F3EAIXux2oQFKm0ihMsOF71dHYx+f3NND68ghCu1YIoePPQN1pGRABkJ6Bus96CutRZMydTl+TvuiRW1m3n0eDl0vRPcEysqdXn+jsQPsrHMquGeXEaY4Yk4wxWcY5V/9scqOMOVUFthatyTy8QyqwZ+kDURKoMWxNKr2EeqVKcTNOajqKoBgOE28U4tdQl5p5bwCw7BWquaZSzAPlwjlithJtp3pTImSqQRrb2Z8PHGigD4RZuNX6JYj6wj7O4TFLbCO/Mn/m8R+h6rYSUb3ekokRY6f/YukArN979jcW+V/S8g0eT/N3VN3kTqWbQ428m9/8k0P/1aIhF36PccEl6EhOcAUCrXKZXXWS3XKd2vc/TRBG9O5ELC17MmWubD2nKhUKZa26Ba2+D3P+4/MNCFwg59oWVeYhkzgN/JDR8deKBoD7Y+ljEjGZ0sosXVTvbc6RHirr2reNy1OXd6pJsQ+gqjk8VWFYmHrwBzW/n+uMPFiRwHB2I7ih8ciHFxIkd/3Omk5tCDV1t+2nNu5sxxpDFNx+huNhVT3/zMDz8usXC3ddaHBj1GHj/As08fwTS7Kt1HBTmyN29vdwAw+/wbwLVOJ3uAD1wi/dUH7Qei66PfyuRj4Ik9is+hglfbkbfR3cnZm7chlUWLdwmprtCohX4HUtlOcQjLYCu+fzGJH2QRKvP3UNz8bWk1qMxjGTOMThZ3kvgLI5AzFfo379UAAAAASUVORK5CYII="
base64img2 = r"iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAApgAAAKYB3X3/OAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAANCSURBVEiJtZZPbBtFFMZ/M7ubXdtdb1xSFyeilBapySVU8h8OoFaooFSqiihIVIpQBKci6KEg9Q6H9kovIHoCIVQJJCKE1ENFjnAgcaSGC6rEnxBwA04Tx43t2FnvDAfjkNibxgHxnWb2e/u992bee7tCa00YFsffekFY+nUzFtjW0LrvjRXrCDIAaPLlW0nHL0SsZtVoaF98mLrx3pdhOqLtYPHChahZcYYO7KvPFxvRl5XPp1sN3adWiD1ZAqD6XYK1b/dvE5IWryTt2udLFedwc1+9kLp+vbbpoDh+6TklxBeAi9TL0taeWpdmZzQDry0AcO+jQ12RyohqqoYoo8RDwJrU+qXkjWtfi8Xxt58BdQuwQs9qC/afLwCw8tnQbqYAPsgxE1S6F3EAIXux2oQFKm0ihMsOF71dHYx+f3NND68ghCu1YIoePPQN1pGRABkJ6Bus96CutRZMydTl+TvuiRW1m3n0eDl0vRPcEysqdXn+jsQPsrHMquGeXEaY4Yk4wxWcY5V/9scqOMOVUFthatyTy8QyqwZ+kDURKoMWxNKr2EeqVKcTNOajqKoBgOE28U4tdQl5p5bwCw7BWquaZSzAPlwjlithJtp3pTImSqQRrb2Z8PHGigD4RZuNX6JYj6wj7O4TFLbCO/Mn/m8R+h6rYSUb3ekokRY6f/YukArN979jcW+V/S8g0eT/N3VN3kTqWbQ428m9/8k0P/1aIhF36PccEl6EhOcAUCrXKZXXWS3XKd2vc/TRBG9O5ELC17MmWubD2nKhUKZa26Ba2+D3P+4/MNCFwg59oWVeYhkzgN/JDR8deKBoD7Y+ljEjGZ0sosXVTvbc6RHirr2reNy1OXd6pJsQ+gqjk8VWFYmHrwBzW/n+uMPFiRwHB2I7ih8ciHFxIkd/3Omk5tCDV1t+2nNu5sxxpDFNx+huNhVT3/zMDz8usXC3ddaHBj1GHj/As08fwTS7Kt1HBTmyN29vdwAw+/wbwLVOJ3uAD1wi/dUH7Qei66PfyuRj4Ik9is+hglfbkbfR3cnZm7chlUWLdwmprtCohX4HUtlOcQjLYCu+fzGJH2QRKvP3UNz8bWk1qMxjGTOMThZ3kvgLI5AzFfo379UAAAAASUVORK5CYII="
base64imgcv = open_image_in_cv(base64img)
base64img2cv = open_image_in_cv(base64img2)
base64imgcv = open_image_in_cv(base64img, channels_in_output=4)
base64img2cv = open_image_in_cv(base64img2, channels_in_output=4)
base64imgcv = open_image_in_cv(base64img, channels_in_output=2)
base64img2cv = open_image_in_cv(base64img2, channels_in_output=2)

#urls
pininterestlogo = "https://camo.githubusercontent.com/7f81f312b05694ccc8cd29e3c3466945ff8e73a13320d3fd0f90c6915bbb4ffb/68747470733a2f2f63646e2e6a7364656c6976722e6e65742f67682f646d68656e647269636b732f7369676e61747572652d736f6369616c2d69636f6e732f69636f6e732f726f756e642d666c61742d66696c6c65642f353070782f70696e7465726573742e706e67"
linkcv1 = open_image_in_cv(pininterestlogo)
linkcv2 = open_image_in_cv(pininterestlogo, channels_in_output=4)
linkcv3 = open_image_in_cv(pininterestlogo, channels_in_output=2)
linkcv4 = open_image_in_cv(pininterestlogo, channels_in_output=3)

#bytes/raw data
byteimage = base64.b64decode(base64img2)
byteimage1 = open_image_in_cv(byteimage)
byteimage2 = open_image_in_cv(byteimage, channels_in_output=4)
byteimage3 = open_image_in_cv(byteimage, channels_in_output=2)
byteimage4 = open_image_in_cv(byteimage, channels_in_output=3)

#PIL
pilimage = Image.fromarray(byteimage2)
pilimage1 = open_image_in_cv(pilimage)
pilimage2 = open_image_in_cv(pilimage, channels_in_output=4)
pilimage3 = open_image_in_cv(pilimage, channels_in_output=2)
pilimage4 = open_image_in_cv(pilimage, channels_in_output=3)

#float images to np.uint8
floatimage = pilimage4.astype(float)
floatimage1 = open_image_in_cv(floatimage)
floatimage2 = open_image_in_cv(floatimage, channels_in_output=4)
floatimage3 = open_image_in_cv(floatimage, channels_in_output=2)
floatimage4 = open_image_in_cv(floatimage, channels_in_output=3)

#filepath
filepath = "c:\\testestestes.png"
pilimage.save(filepath)
filepath1 = open_image_in_cv(filepath, bgr_to_rgb=True)
filepath2 = open_image_in_cv(filepath, channels_in_output=4, bgr_to_rgb=True)
filepath3 = open_image_in_cv(filepath, channels_in_output=2, bgr_to_rgb=True)
filepath4 = open_image_in_cv(filepath, channels_in_output=3, bgr_to_rgb=True)


from a_cv2_imshow_thread import add_imshow_thread_to_cv2
add_imshow_thread_to_cv2()
cv2.imshow_thread(
    [
        base64imgcv,
        base64img2cv,
        linkcv1,
        linkcv2,
        linkcv3,
        linkcv4,
        byteimage1,
        byteimage2,
        byteimage3,
        byteimage4,
        pilimage1,
        pilimage2,
        pilimage3,
        pilimage4,
        floatimage1,
        floatimage2,
        floatimage3,
        floatimage4,
        filepath1,
        filepath2,
        filepath3,
        filepath4,
    ]
)
        Parameters:
            image:Any
                Can be URL, bytes, base64, file path, np.ndarray, PIL
            channels_in_output:Union[int,None]
                None (original image won't be changed)
                2 (GRAY)
                3 (BGR)
                4 (BGRA)
                (default=None)
            bgr_to_rgb:bool=False
                Converts BGRA to RGBA / BGR to RGB
        Returns:
            image:np.ndarray
                (Always as np.uint8)
```
