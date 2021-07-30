This project is an extension of the excellent face_recognition library for my personal needs. It uses face_recognition's CNN function to extract the faces from an image. Because not every image fits in my GPU's memory as-is, the image is scaled down to have the number of pixels defined in parameters.xml, namely the product of 'height' and 'width'. However, the image isn't resized to those values; it keeps the aspect ratio by computing what the width and height should be to get approximately that number of pixels.

The code also extracts the faces at smaller, pyramidal levels at the same resolution. In effect, this extracts faces at the bigger pixel, then zooms in on each quadrant of the image and extracts those faces at a higher resolution. The code then fuses the two levels of detections cohesively by looking at bounding rectangles and face encodings. 

The code is also set up with a server-client module. Given an image on the client side, it is converted into a bit stream and sent to the server, where the processing is completed. Code is also provided to connect a client with multiple servers, in case large numbers of images need to be processed. 

It may be desirable to set the server code to run on boot for a given device, such as a Nvidia Jetson Nano. The following instructions describe how to do this on a Linux system. This project is not yet tested on Windows or Mac.

==Server Installation==
Assumptions: The server will run on port 5000. This port is defined in parameters.xml; for the purposes of this instruction, 5000 will be used.

- Use ufw (universal firewall) to enable communication on port 5000:
    `sudo ufw enable`
    `sudo ufw allow 5000`
- Find the path to your gunicorn installation, using `which gunicorn`. Given that path, put the following in your user space crontab: 
    `@reboot (<WHICH_GUNICORN_OUTPUT> -b 0.0.0.0:5000 -w 1 --chdir <PATH_TO_THIS_GIT_REPO> server_client.server_image_handler:app --timeout 120 ) &`
    (The timeout option is to allow the worker ample time to process the image)
- Set the server_ip file to run on boot in crontab. The sleep is to enable the networking to iron itself out. 
    `@reboot (sleep 15; /usr/bin/python3 <PATH_TO_THIS_GIT_REPO>/server_client/server_ip_discover.py) &`
