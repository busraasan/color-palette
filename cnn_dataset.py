from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import skimage.color as scicolor
from utils import *

class PreviewDataset(Dataset):
    def __init__(self, root="../destijl_dataset/rgba_dataset/", 
                 transform=None, test=False, color_space="RGB", 
                 input_color_space="RGB", 
                 is_classification=False, 
                 normalize_cielab=True,
                 normalize_rgb=True):

        self.test = test
        self.sample_filenames = os.listdir(root+"00_preview_cropped")
        self.transform = transform
        self.img_dir = root
        self.color_space = color_space
        self.is_classification = is_classification
        self.input_color_space = input_color_space
        self.normalize_cielab = normalize_cielab
        self.normalize_rgb = normalize_rgb

        self.train_filenames, self.test_filenames = train_test_split(self.sample_filenames,
                                                                     test_size=0.2, 
                                                                     random_state=42) 
    def __len__(self):
        if self.test:
            return len(self.test_filenames)
        else:
            return len(self.train_filenames)
        
    def __getitem__(self, idx):

        path_idx = "{:04d}".format(idx)
        img_path = os.path.join(self.img_dir, "00_preview_cropped/" + self.sample_filenames[idx])

        image = np.array(Image.open(img_path))
        # Convert image to lab if the input space is CIELab.
        # Image is a numpy array always. Convert to tensor at the end.
        if self.input_color_space == "CIELab":
            image = scicolor.rgb2lab(image)
            if self.normalize_cielab:
                image = torch.from_numpy(image)
                image = normalize_CIELab(image)
        else:
            image = torch.from_numpy(image)
        
        # Apply kmeans on RGB image always.
        bg_path = os.path.join("../destijl_dataset/01_background/" + self.sample_filenames[idx])
        # Most dominant color in RGB.
        color = self.kmeans_for_bg(bg_path)[0]

        # If output is in CIELab space but input is in RGB, convert target to CIELab also.
        if self.color_space == "CIELab" and self.input_color_space == "RGB":
            target_color = torch.squeeze(torch.tensor(RGB2CIELab(color.astype(np.int32))))
            if self.normalize_cielab:
                target_color = normalize_CIELab(target_color)
        # Input and output is in RGB space or input and output is in CIELab space.
        # If Input is in CIELab and output is in RGB, than this is also valid since dataset is in RGB.
        else:
            target_color = torch.squeeze(torch.tensor(color))

        if self.is_classification:
            target_color = [torch.zeros(256), torch.zeros(256), torch.zeros(256)]
            target_color[0][color[0]] = 1
            target_color[1][color[1]] = 1
            target_color[2][color[2]] = 1

        if self.transform:
            # Reshape the image if not in (C, H, W) form.
            if image.shape[0] != 3:
                image = image.reshape(-1, image.shape[0], image.shape[1]).type("torch.FloatTensor")
            # Apply the transformation
            image = self.transform(image)

        if self.color_space == "RGB" and self.normalize_rgb:
            target_color /= 255

        return image, target_color
    
    def kmeans_for_bg(self, bg_path):
        image = cv2.imread(bg_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        n_colors = 1

        # Apply KMeans to the text area
        pixels = np.float32(image.reshape(-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS

        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        palette = np.asarray(palette, dtype=np.int64) # RGB

        return palette