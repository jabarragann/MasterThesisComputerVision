import cv2
import numpy as np

if __name__ == "__main__":
    cap1 = cv2.VideoCapture('./videos/video_right_color_session-3-r.avi')
    cap2 = cv2.VideoCapture('./videos/result/video_right_color_session-3-r.avi_processed.avi')

    # Check if camera opened successfully
    if not cap1.isOpened() or not cap2.isOpened():
      print("Error opening video stream or file")

    #Output video
    resultPath = './videos/result/together.avi'
    frame_width = 640*2
    frame_height = 480
    out = cv2.VideoWriter(str(resultPath), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 60,
                        (frame_width, frame_height))

    while cap1.isOpened():
      ret, frame1 = cap1.read() # Capture frame-by-frame
      ret, frame2 = cap2.read() # Capture frame-by-frame

      if ret:
        finalFrame = np.hstack((frame1, frame2))
        out.write(finalFrame)
        cv2.imshow('Frame',finalFrame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
      else:
        break

    # When everything done, release the video capture object
    cap1.release()
    cap2.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()