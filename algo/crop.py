import cv2
import os
def algo(filename):
  path=r'C:\Users\HP\Downloads\Doctors Prescriptions Final Edit\project final'
  folder=os.listdir(path)
  for i in folder:
      print(i)
      if i.endswith('.jpg')==True:
          filename=os.path.join(path,i)
          os.remove(filename)
          
  def crop_image(img):
    
    ref_point = []
    cropping = False

    def shape_selection(event, x, y, flags, param):

      nonlocal ref_point, cropping

      
      if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

      
      elif event == cv2.EVENT_LBUTTONUP:
        
        ref_point.append((x, y))
        cropping = False

        
        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("image", image)
    count=1
    while True:

      image=cv2.imread(img,0)
      image=cv2.resize(image,(700,700))
      
    
      clone = image.copy()
      cv2.namedWindow("image")
      cv2.setMouseCallback("image", shape_selection)
      print('Now crop image. press "r" to reset cropping.press "c" to confirm cropping')
    
    
      
      cv2.imshow("image", image)
      
      key = cv2.waitKey(1) & 0xFF

      
      if key == ord("r"):
        image = clone.copy()
        

      
      elif key == ord("c"):

        print('crop')
      
    
        if len(ref_point) == 2:
          crop_img = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
          print(crop_img.shape)

          cv2.imshow("crop_img", crop_img)
          
          filename = 'croped image'+str(count)+'.jpg'

          print(os.path.join(path,filename))
          cv2.imwrite(os.path.join(r'C:\Users\HP\Downloads\Doctors Prescriptions Final Edit\project final',filename),crop_img)
          count=count+1

        cv2.destroyAllWindows()

      elif key == ord("s"):
          break

  cwd=os.getcwd()
  path = os.path.join(cwd,'media','Prescription',filename)
  
  crop_image(path)
  print("in cropp end")
