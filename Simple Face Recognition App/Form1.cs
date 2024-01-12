using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Face;
using Emgu.CV.CvEnum;
using System.IO;
using System.Threading;
using System.Diagnostics;

namespace Simple_Face_Recognition_App
{
    public partial class Form1 : Form
    {
        // First !!!
        // Go to Solution Explorer and Right Click,
        // Click on "Restore NuGet Packages"
        // Run the application

        #region Variables
        int testid = 0;
        private Capture videoCapture = null; // needed for using WebCam
        private Image<Bgr, Byte> currentFrame = null; // used for analysing frame/image for faces
        Mat frame = new Mat(); // unchanged frame (needed for conversion to usable currentFrame)
        private bool facesDetectionEnabled = false; // used to decide if face should be detected
        private static string trainedDataFile = @"C:\Users\DominikG\source\repos\Simple-Face-Recognition-App-CS\Simple Face Recognition App\bin\Debug\haarcascade_frontalface_alt.xml"; // Backup of trained Data
        CascadeClassifier faceCasacdeClassifier = new CascadeClassifier(trainedDataFile); // Colletion of trained Data
        Image<Bgr, Byte> faceResult = null; // used to convert Capture/Images from normal Color Code to Bgr
        List<Image<Gray, Byte>> TrainedFaces = new List<Image<Gray, byte>>(); // Collection of (while Runtime) trained and identified Faces
        List<int> PersonsLabes = new List<int>(); // used for transfering training data to Collection

        bool EnableSaveImage = false; // used to decide if images should get saved
        private bool  isTrained = false; // change status if face is treined
        EigenFaceRecognizer recognizer; // instance of facial detection library
        List<string> PersonsNames = new List<string>();

        #endregion

        public Form1()
        {
            InitializeComponent();
        }

        private void btnCapture_Click(object sender, EventArgs e) // activate camers
        {
            //Dispose of Capture if it was created before
            if (videoCapture != null) videoCapture.Dispose(); 
            
            videoCapture = new Capture();
            //videoCapture.ImageGrabbed += ProcessFrame;
            Application.Idle += ProcessFrame; 
            // videoCapture.Start();
        }

        
        private void ProcessFrame(object sender, EventArgs e) // used to process the Frames
        {
            //Step 1: Video Capture
            // if no current Capture is acitve
            if (videoCapture != null && videoCapture.Ptr != IntPtr.Zero) 
            {
                // start Capture
                videoCapture.Retrieve(frame, 0); 
                
                // convert frame to usable image => currentFrame
                currentFrame = frame.ToImage<Bgr, Byte>().Resize(picCapture.Width, picCapture.Height, Inter.Cubic); 

                //Step 2: Face Detection
                if (facesDetectionEnabled)
                {

                    //Convert from Bgr to Gray Image
                    Mat grayImage = new Mat();
                    CvInvoke.CvtColor(currentFrame, grayImage, ColorConversion.Bgr2Gray);

                    //Enhance the image to get better result
                    CvInvoke.EqualizeHist(grayImage, grayImage);

                    // detect Face
                    Rectangle[] faces = faceCasacdeClassifier.DetectMultiScale(grayImage, 1.1, 3, Size.Empty, Size.Empty);

                    // If faces detected
                    if (faces.Length > 0)
                    {

                        foreach (var face in faces) // process each frame
                        {
                            //Draw square around each face 
                           // CvInvoke.Rectangle(currentFrame, face, new Bgr(Color.Red).MCvScalar, 2);

                            //Step 3: Add Person 
                            //Assign the face to the picture Box face picDetected
                            Image<Bgr, Byte> resultImage = currentFrame.Convert<Bgr, Byte>();
                            resultImage.ROI = face;
                            // get zoomed Capture of detected face to better show what got detected
                            picDetected.SizeMode = PictureBoxSizeMode.StretchImage; 
                            picDetected.Image = resultImage.Bitmap;

                            if (EnableSaveImage)
                            {
                                //We will create a directory if does not exists
                                string path = Directory.GetCurrentDirectory() + @"\TrainedImages";
                                if (!Directory.Exists(path))
                                    Directory.CreateDirectory(path);

                                //we will save 10 images with delay a second for each image 
                                //to avoid hang GUI we will create a new task
                                Task.Factory.StartNew(() => {
                                    for (int i = 0; i < 10; i++)
                                    {
                                        //resize the image then saving it
                                        resultImage.Resize(200, 200, Inter.Cubic).Save(path + @"\" + txtPersonName.Text +"_"+ DateTime.Now.ToString("dd-mm-yyyy-hh-mm-ss") + ".jpg");
                                        Thread.Sleep(1000);
                                    }
                                });

                            }
                            EnableSaveImage = false;

                            // force activation to allow saving of names to a person
                            if (btnAddPerson.InvokeRequired)
                            {
                                btnAddPerson.Invoke(new ThreadStart(delegate {
                                    btnAddPerson.Enabled = true;
                                }));
                            }

                            // Step 5: Recognize the face 
                            if (isTrained)
                            {
                                // convert to Gray image
                                Image<Gray, Byte> grayFaceResult = resultImage.Convert<Gray, Byte>().Resize(200,200,Inter.Cubic);
                                
                                // enhance quality for easyer detection
                                CvInvoke.EqualizeHist(grayFaceResult,grayFaceResult); 
                                
                                // detect face
                                var result = recognizer.Predict(grayFaceResult);
                                
                                // save image to picture box for better displaying what got detected
                                pictureBox1.Image = grayFaceResult.Bitmap;
                                pictureBox2.Image = TrainedFaces[result.Label].Bitmap;
                                Debug.WriteLine(result.Label+". "+result.Distance);

                                
                                if (result.Label != -1 && result.Distance < 2000)//Here results found known faces
                                {
                                    // save found data to the trained data 
                                    CvInvoke.PutText(currentFrame, PersonsNames[result.Label], new Point(face.X - 2, face.Y - 2),
                                        FontFace.HersheyComplex, 1.0, new Bgr(Color.Orange).MCvScalar);
                                    CvInvoke.Rectangle(currentFrame, face, new Bgr(Color.Green).MCvScalar, 2);
                                }
                                else //here results did not found any know faces
                                {
                                    // save as Unknown
                                    CvInvoke.PutText(currentFrame, "Unknown", new Point(face.X - 2, face.Y - 2),
                                        FontFace.HersheyComplex, 1.0, new Bgr(Color.Orange).MCvScalar);
                                    CvInvoke.Rectangle(currentFrame, face, new Bgr(Color.Red).MCvScalar, 2);

                                }
                            }
                        }
                    }
                }

                //Render the video capture into the Picture Box picCapture
                picCapture.Image = currentFrame.Bitmap;
            }

            //Dispose the Current Frame after processing it to reduce the memory consumption.
            if (currentFrame != null)
                currentFrame.Dispose();
        }

        private void btnDetectFaces_Click(object sender, EventArgs e)
        {
            facesDetectionEnabled = true;
        }

        private void btnAddPerson_Click(object sender, EventArgs e)
        {
            btnAddPerson.Enabled = false;
            EnableSaveImage = true;
        }

        private void btnTrain_Click(object sender, EventArgs e)
        {
            TrainImagesFromDir();
        }
        //Step 4: train Images .. we will use the saved images from the previous example 
        private bool TrainImagesFromDir()
        {
            int ImagesCount = 0;
            double Threshold = 2000;
            TrainedFaces.Clear();
            PersonsLabes.Clear();
            PersonsNames.Clear();

            try
            {
                // get Directorys
                string path = Directory.GetCurrentDirectory() + @"\TrainedImages";
                string[] files = Directory.GetFiles(path, "*.jpg", SearchOption.AllDirectories);

                foreach (var file in files) // relearn old backuped data
                {
                    Image<Gray, byte> trainedImage = new Image<Gray, byte>(file).Resize(200,200,Inter.Cubic);
                    CvInvoke.EqualizeHist(trainedImage,trainedImage); // enhance image quality for better detection
                    TrainedFaces.Add(trainedImage);
                    PersonsLabes.Add(ImagesCount);
                    string name = file.Split('\\').Last().Split('_')[0]; 
                    PersonsNames.Add(name);
                    ImagesCount++;
                    Debug.WriteLine(ImagesCount + ". " +name);

                }

                if (TrainedFaces.Count() > 0)
                {
                    // recognizer = new EigenFaceRecognizer(ImagesCount,Threshold);
                    recognizer = new EigenFaceRecognizer(ImagesCount, Threshold);
                    recognizer.Train(TrainedFaces.ToArray(), PersonsLabes.ToArray());

                    isTrained = true;
                    //Debug.WriteLine(ImagesCount);
                    //Debug.WriteLine(isTrained);
                    return true;
                }
                else
                {
                    isTrained = false;
                    return false;
                }
            }
            catch (Exception ex)
            {
                isTrained = false;                
                MessageBox.Show("Error in Train Images: " + ex.Message);
                return false;
            }
            
        }
       
    }
}
