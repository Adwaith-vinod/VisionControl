import vlc
import time


class VLCController:

    def __init__(self, video_path):
        self.player = vlc.MediaPlayer(video_path)
        self.last_gesture = None

    def play(self):
        self.player.play()

    def pause(self):
        self.player.pause()

    def stop(self):
        self.player.stop()

    def handle_gesture(self, gesture):

        # Ignore repeated predictions
        if gesture == self.last_gesture:
            return

        # Store the new gesture
        self.last_gesture = gesture

        # Fist -> Pause
        if gesture == "fist":
            print("Gesture: FIST -> PAUSE")
            self.pause()

        # Open hand -> Play
        elif gesture == "open":
            print("Gesture: OPEN -> PLAY")
            self.play()

    def close(self):
        self.player.stop()


if __name__ == "__main__":

    controller = VLCController("video.mp4")

    print("Playing video...")
    controller.play()

    time.sleep(5)

    print("Pausing...")
    controller.handle_gesture("fist")

    time.sleep(3)

    print("Playing again...")
    controller.handle_gesture("open")

    time.sleep(5)

    controller.close()

    print("VLC test finished.")