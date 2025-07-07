import torch
import torch.nn.functional as F
import torch.utils.data
import torch.backends.cudnn as cudnn
import string

from dl_ocr_bench.utils import CTCLabelConverter, AttnLabelConverter
from dl_ocr_bench.dataset import RawDataset, AlignCollate
from dl_ocr_bench.model import Model


class Predictor:
    """A class for handling text recognition inference with pretrained models."""
    
    def __init__(self, opt):
        """Initialize the predictor with model configuration."""
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup character converter
        if 'CTC' in opt.Prediction:
            self.converter = CTCLabelConverter(opt.character)
        else:
            self.converter = AttnLabelConverter(opt.character)
        opt.num_class = len(self.converter.character)
        
        # Configure input channels
        if opt.rgb:
            opt.input_channel = 3
        
        # Initialize model
        self.model = Model(opt)
        print('Model input parameters:', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, 
              opt.output_channel, opt.hidden_size, opt.num_class, opt.batch_max_length, 
              opt.Transformation, opt.FeatureExtraction, opt.SequenceModeling, opt.Prediction)
        
        # Load model to device
        self.model = torch.nn.DataParallel(self.model).to(self.device)
        
        # Load pretrained weights
        print('Loading pretrained model from %s' % opt.saved_model)
        self.model.load_state_dict(torch.load(opt.saved_model, map_location=self.device))
        self.model.eval()
        
        # Setup data loading
        self.align_collate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        
        # Configure CUDNN
        cudnn.benchmark = True
        cudnn.deterministic = True
    
    def create_data_loader(self, image_folder):
        """Create data loader for the given image folder."""
        demo_data = RawDataset(root=image_folder, opt=self.opt)
        demo_loader = torch.utils.data.DataLoader(
            demo_data, 
            batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=int(self.opt.workers),
            collate_fn=self.align_collate, 
            pin_memory=True
        )
        return demo_loader
    
    def predict_batch(self, image_tensors):
        """Predict text for a batch of image tensors."""
        with torch.no_grad():
            batch_size = image_tensors.size(0)
            image = image_tensors.to(self.device)
            
            # Setup prediction tensors
            length_for_pred = torch.IntTensor([self.opt.batch_max_length] * batch_size).to(self.device)
            text_for_pred = torch.LongTensor(batch_size, self.opt.batch_max_length + 1).fill_(0).to(self.device)
            
            # Forward pass
            if 'CTC' in self.opt.Prediction:
                preds = self.model(image, text_for_pred)
                # Select max probability (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                preds_str = self.converter.decode(preds_index, preds_size)
            else:
                preds = self.model(image, text_for_pred, is_train=False)
                # Select max probability (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = self.converter.decode(preds_index, length_for_pred)
            
            # Calculate confidence scores
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            
            # Process predictions
            results = []
            for pred, pred_max_prob in zip(preds_str, preds_max_prob):
                if 'Attn' in self.opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]
                
                # Calculate confidence score (multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                results.append((pred, confidence_score.item()))
            
            return results
    
    def predict_images(self, image_folder):
        """Predict text for all images in the given folder."""
        data_loader = self.create_data_loader(image_folder)
        all_results = []
        
        for image_tensors, image_path_list in data_loader:
            batch_results = self.predict_batch(image_tensors)
            
            # Combine results with image paths
            for img_path, (pred_text, confidence) in zip(image_path_list, batch_results):
                all_results.append({
                    'image_path': img_path,
                    'predicted_text': pred_text,
                    'confidence_score': confidence
                })
        
        return all_results