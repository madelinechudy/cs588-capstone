# cs588-capstone


# RESNET TRAINING

Run py resnet_training.py to start training the model. The program will train with 10 epochs, 
reporting the accuracy and loss after each. Then the pretrained model will be saved in 
"../cs588-capstone/segmentation/models".

FOR THE UI: 

Start the Flask backend (python app.py).
Start the React frontend (npm start).
Open the React app in your browser, upload an MRI image, and click Submit.
The React app will send the image to the Flask server, which will run the prediction and send the result back to the frontend to display.

RUNNING app.py
python src/app.py --unet_model_path="C:\Users\maddi\Documents\cs588-capstone\Segmentation\Models\pretrained_unet_model.keras" `
                  --resnet_model_path="C:\Users\maddi\Documents\cs588-capstone\Segmentation\Models\resnet_model.keras"


RUNNING Predict.py
python predict.py --test_images_dir "C:\Users\maddi\Documents\cs588-capstone\Data\Testing" --resnet_model_path "C:\Users\maddi\Documents\cs588-capstone\Segmentation\Models\ResNet\Classification\resnet_model.keras" --batch_size 32