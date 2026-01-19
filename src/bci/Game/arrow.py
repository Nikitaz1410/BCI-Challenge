import sys
import os
import csv
import random
import yaml
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QLineEdit, QComboBox,
                             QPushButton, QWidget, QMessageBox, QSlider, QStackedWidget, QGroupBox, QRadioButton, QButtonGroup)

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor
from pylsl import StreamInfo, StreamOutlet
from PyQt5.QtMultimedia import QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QUrl


class CrossDotWidget(QWidget):
    def paintEvent(self, event):
        painter = QPainter(self)
        if not painter.isActive():
            return

        painter.setRenderHint(QPainter.Antialiasing)

        center_x = self.width() // 2
        center_y = self.height() // 2

        pen = QPen(QColor(0, 0, 0), 4)
        painter.setPen(pen)
        painter.drawLine(center_x - 40, center_y, center_x + 40, center_y)
        painter.drawLine(center_x, center_y - 40, center_x, center_y + 40)

        painter.setBrush(QColor(0, 0, 0))
        painter.drawEllipse(center_x - 5, center_y - 5, 10, 10)
        
class SurveyApp(QMainWindow):
    """GUI Application for a survey and an arrow-based stimulus paradigm."""

    def __init__(self, cwdPath, part_number, cur_session):
        """Initialize the survey application with GUI, LSL, and configuration settings."""
        super().__init__()
        self.setWindowTitle("Survey Application")

        self.cwdPath = cwdPath
        self.part_number = part_number
        self.cur_session = cur_session
        

        # Initialize LSL stream
        self.lsl_info = StreamInfo('Markers', 'Markers', 1, 0, 'string', 'arrowParadigm12345')
        self.lsl_outlet = StreamOutlet(self.lsl_info)
        
        # setup reactiongame
        self.reactiongame_started = False

        # Survey data storage
        self.responses = {}
        self.current_page = 0
        self.current_run = 0
        self.current_trial = 0
        self.stimuli_sequence = []
        self.in_arrow_paradigm = False
        self.break_questions_start = False
        self.questions_len = 0
        self.current_stimulus = None


        # Load settings from config
        config_path = os.path.join(cwdPath, "config", "arrow_config.yaml")
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)


        input_lang = input("Enter the language of the survey: 'eng' (default), 'de': ")
        if input_lang == "":
            input_lang = "eng"
        print(f"Using language: {input_lang}")
        self.questions = config[f'questions_{input_lang}']
        self.break_questions = config[f'break_questions_{input_lang}']
        
        self.stimulus_duration = int(config['stimulus_duration'])
        self.cross_duration = int(config['cross_duration'])
        self.black_screen_duration = int(config['black_screen_duration'])
        self.num_trials = int(config['num_trials'])
        self.num_repetitions = int(config['num_repetitions'])
        self.run_start_pause = int(config['run_start_pause'])
        self.config = config
        self.num_max_runs = int(config['num_max_runs'])

        # Load images
        assets_folder = os.path.join(cwdPath, "assets")
        self.images = {
            "left": os.path.join(assets_folder, config["images"]["left"]),
            "right": os.path.join(assets_folder, config["images"]["right"]),
            "circle": os.path.join(assets_folder, config["images"]["circle"]),
            "cross": os.path.join(assets_folder, config["images"]["cross"]),
        }

        # Setup UI
        self.central_widget = QWidget()
        #self.setFixedSize(800, 800)
        self.setCentralWidget(self.central_widget)
        self.central_widget.setStyleSheet("background-color: #000000; color: white; font-family: Arial;")
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setSpacing(40)

        self.question_label = QLabel()
        self.question_label.setAlignment(Qt.AlignCenter)
        self.question_label.setWordWrap(True)
        self.question_label.setStyleSheet("font-size: 36px; margin-bottom: 20px;")
        self.layout.addWidget(self.question_label)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        self.input_widget = None
        self.slider_label = None

        self.nav_layout = QHBoxLayout()
        self.nav_layout.setSpacing(60)
        self.prev_button = QPushButton("Previous")
        self.prev_button.setStyleSheet("font-size: 28px; height: 100px; width: 300px; background-color: darkgray; border-radius: 15px;")
        self.prev_button.clicked.connect(self.previous_page)
        self.nav_layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next")
        self.next_button.setStyleSheet("font-size: 28px; height: 100px; width: 300px; background-color: darkgray; border-radius: 15px;")
        self.next_button.clicked.connect(self.next_page)
        self.nav_layout.addWidget(self.next_button)
        # Cross screen
        self.cross_screen = CrossDotWidget()
        
        self.submit_button = QPushButton("Submit")
        self.submit_button.setStyleSheet("font-size: 28px; height: 100px; width: 300px; background-color: green; color: white; border-radius: 15px;")
        self.submit_button.clicked.connect(self.submit)
        self.submit_button.hide()
        self.nav_layout.addWidget(self.submit_button)

        self.layout.addLayout(self.nav_layout)

        self.start_button = QPushButton("Start Arrow Paradigm")
        self.start_button.setStyleSheet("font-size: 36px; height: 120px; width: 400px; background-color: red; color: white; border-radius: 20px;")
        self.start_button.clicked.connect(self.start_arrow_paradigm)
        self.start_button.hide()
        self.layout.addWidget(self.start_button, alignment=Qt.AlignCenter)

        self.start_reactiontest_button = QPushButton("Start Reaction Test")
        self.start_reactiontest_button.setStyleSheet("font-size: 36px; height: 120px; width: 400px; background-color: blue; color: white; border-radius: 20px;")
        self.start_reactiontest_button.clicked.connect(self.start_reaction_time)
        self.start_reactiontest_button.hide()
        self.layout.addWidget(self.start_reactiontest_button, alignment=Qt.AlignCenter)
        
        self.click_button = QPushButton("Click!")
        self.click_button.setStyleSheet("font-size: 36px; height: 220px; width: 600px; background-color: black; color: white; border-radius: 20px;")
        self.click_button.clicked.connect(self.reaction_click)
        self.click_button.hide()
        self.layout.addWidget(self.click_button, alignment=Qt.AlignCenter)
        
        self.accept_reaction_button = QPushButton("Save reaction time") 
        self.accept_reaction_button.setStyleSheet("font-size: 36px; height: 120px; width: 400px; background-color: green; color: white; border-radius: 20px;")
        self.accept_reaction_button.clicked.connect(self.save_reaction_time)
        self.accept_reaction_button.hide()
        self.layout.addWidget(self.accept_reaction_button, alignment=Qt.AlignCenter)
        self.load_page()

    def reset_widgets(self):
        """Reset UI widgets to prepare for the next survey load."""
        self.question_label.show()
        self.image_label.hide()
        self.start_button.hide()
        self.start_reactiontest_button.hide()
        if self.input_widget:
            self.input_widget.hide()
        if self.slider_label:
            self.slider_label.hide()
        self.submit_button.show()
        self.prev_button.show()
        self.next_button.show()
        

    def load_page(self):
        """Load the current page's question and input widget."""
        self.reset_widgets()
        if self.break_questions_start:
            current_question = self.break_questions[self.current_page]
            self.questions_len = len(self.break_questions)
        else:
            current_question = self.questions[self.current_page]
            self.questions_len = len(self.questions)
            
        if self.current_page < self.questions_len:
            self.question_label.setText(current_question["question"])

            question_type = current_question["type"]
            
            if question_type == "entry":
                self.input_widget = QLineEdit()
                get_default = current_question.get("default", "")
                self.input_widget.setText(get_default)
                self.input_widget.setStyleSheet(
                    "font-size: 28px; height: 60px; width: 600px; background-color: grey; color: black; border-radius: 15px; padding: 10px;"
                )
                self.input_widget.setAlignment(Qt.AlignCenter)
            elif question_type == "options":
                # --- container that just frames the radios ---
                self.input_widget = QGroupBox()
                self.input_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.input_widget.setStyleSheet(
                    "QGroupBox {background:#111; color:white; border-radius:10px; padding:15px;}"
                    "QRadioButton {font-size:24px; height:50px; width:500px; "
                    "background:grey; color:black; border-radius:10px; padding:5px;}"
                    # highlight the one that is selected
                    "QRadioButton::indicator:checked {background:#ffcc00;}"
                )

                # --- vertical layout INSIDE the group box ---
                vbox = QVBoxLayout(self.input_widget)
                vbox.setSpacing(20)
                vbox.setAlignment(Qt.AlignmentFlag.AlignCenter)

                # --- *exclusive* button group that owns the radios ---
                self.btn_group = QButtonGroup(self)       # parent keeps it alive
                self.btn_group.setExclusive(True)

                # build one radio for every answer
                answers = current_question.get("set_answers", [])
                for index, answer in enumerate(answers, start=1):
                    rb = QRadioButton(answer)
                    self.btn_group.addButton(rb, index)   # store an id for later
                    vbox.addWidget(rb)

                # optional: select the first item by default
                if self.btn_group.buttons():
                    self.btn_group.buttons()[0].setChecked(True)

                # react whenever the user changes selection
                self.btn_group.idToggled.connect(self.on_option_chosen)

                # finally add the group box to the main page layout
                self.layout.addWidget(self.input_widget, alignment=Qt.AlignmentFlag.AlignCenter)

                    
                
            elif question_type == "range":
                self.input_widget = QSlider(Qt.Horizontal)
                min_value, max_value = current_question.get("min", 1), current_question.get("max", 5)
                self.input_widget.setRange(min_value, max_value)
                self.input_widget.setTickPosition(QSlider.TicksBelow)
                self.input_widget.setTickInterval(1)
                self.input_widget.setStyleSheet(
                    "height: 60px; width: 900px; background-color: #111; color: white; border-radius: 5px; padding: 6px;"
                )
                self.input_widget.setValue((min_value + max_value) // 2)  # Default to midpoint
                self.input_widget.valueChanged.connect(self.update_slider_label)

                # Create slider label
                self.slider_label = QLabel(f"Value: {self.input_widget.value()}")
                self.slider_label.setAlignment(Qt.AlignCenter)
                self.slider_label.setStyleSheet("font-size: 24px; color: #ffcc00; margin-top: 10px;")
                self.layout.insertWidget(2, self.slider_label, alignment=Qt.AlignCenter)
            elif question_type == "range-text":
                self.input_widget = QSlider(Qt.Horizontal)
                min_value, max_value = current_question.get("min", 1), current_question.get("max", 5)
                self.input_widget.setRange(min_value, max_value)
                self.input_widget.setTickPosition(QSlider.TicksBelow)
                self.input_widget.setTickInterval(1)
                self.input_widget.setStyleSheet(
                    "height: 60px; width: 900px; background-color: #111; color: white; border-radius: 5px; padding: 6px;"
                )
                self.input_widget.setValue((min_value + max_value) // 2)  # Default to midpoint
                self.input_widget.valueChanged.connect(self.update_slider_label_text)

                # Create slider label
                self.answer_set = current_question.get("set_answers", [])
                showtext = self.answer_set[self.input_widget.value()-1]
                self.slider_label = QLabel(f"Value: {showtext}")
                self.slider_label.setAlignment(Qt.AlignCenter)
                self.slider_label.setStyleSheet("font-size: 24px; color: #ffcc00; margin-top: 10px;")
                self.layout.insertWidget(2, self.slider_label, alignment=Qt.AlignCenter)

            self.layout.insertWidget(2, self.input_widget, alignment=Qt.AlignCenter)

        # Handle image display
        if "image" in current_question and current_question["image"]:
            image_path = os.path.join(self.cwdPath, "assets", current_question["image"])
            if os.path.exists(image_path):
                pixmap = QPixmap(image_path).scaled(1200, 900, Qt.KeepAspectRatio)
                self.image_label.setPixmap(pixmap)
                self.image_label.show()
            else:
                QMessageBox.critical(self, "Error", f"Image file not found: {image_path}")
                self.image_label.hide()
        else:
            self.image_label.hide()
        # Update navigation buttons
        self.prev_button.setEnabled(self.current_page > 0)
        self.next_button.setVisible(self.current_page < self.questions_len - 1)
        self.submit_button.setVisible(self.current_page == self.questions_len - 1)

    def update_slider_label(self):
        """Update the label displaying the slider's value."""
        if self.slider_label and isinstance(self.input_widget, QSlider):
            self.slider_label.setText(f"Value: {self.input_widget.value()}")
            
    def update_slider_label_text(self):
        """Update the label displaying the slider's value."""
        if self.slider_label and isinstance(self.input_widget, QSlider):
            showtext = self.answer_set[self.input_widget.value()-1]
            self.slider_label.setText(f"Value: {showtext}")
    # elsewhere in your class
    def on_option_chosen(self, btn_id: int, checked: bool):
        if checked:                                  # only act on 'True' edges
            chosen_text = self.btn_group.button(btn_id).text()
            # show the choice in the group-box title (like your lambda used to do)
            self.input_widget.setTitle(f"Selected: {chosen_text}")
            # store / process the answer
            self.current_answer = chosen_text
    def save_response(self):
        """Save the current page's response."""
        if self.current_page < self.questions_len:
            if isinstance(self.input_widget, QLineEdit):
                response = self.input_widget.text()
            elif isinstance(self.input_widget, QComboBox):
                response = self.input_widget.currentText()
            elif isinstance(self.input_widget, QSlider):
                response = str(self.input_widget.value())
            elif isinstance(self.input_widget, QGroupBox):
                selected_button = self.btn_group.checkedButton()
                if selected_button:
                    response = selected_button.text()
                else:
                    response = ""
            else:
                response = ""

            if not response.strip():
                QMessageBox.warning(self, "Error", "Please provide an answer before proceeding.")
                return False

            self.responses[self.current_page] = response
            return True
        return True

    def next_page(self):
        """Go to the next page."""
        if self.save_response():
            self.current_page += 1
            self.load_page()

    def previous_page(self):
        """Go to the previous page."""
        self.current_page -= 1
        self.load_page()

    def submit(self):
        """Submit the survey and save responses to CSV."""
        if not self.save_response():
            return

        self.lsl_outlet.push_sample([self.config["Markers"]["survey_end"]])
        print("Marker sent: SURVEY END")

        try:
            # Save responses to CSV
            participant_id = self.part_number
            current_session = self.cur_session
            Pnum = f"sub-P{str(participant_id).zfill(3)}"
            Snum = f"ses-S{str(current_session).zfill(3)}"
            Rnum = f"run-{str(self.current_run).zfill(3)}"
            folder_path = os.path.join(self.cwdPath, "data", Pnum, Snum)
            os.makedirs(folder_path, exist_ok=True)
            save_file = os.path.join(folder_path, f"{Pnum}_{Snum}_task-arrow_survey_{Rnum}.csv")
            print(f"Saving survey results to {save_file}")
            send_str = ""
            with open(save_file, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                if self.break_questions_start:
                    writer.writerow([q["question"] for q in self.break_questions])
                    answers = [self.responses.get(i, "") for i in range(len(self.break_questions))]
                    questions_short = [q.get("short", "") for q in self.break_questions]

                    self.lsl_outlet.push_sample([",".join(questions_short).upper()])
                    print("Marker sent: " + ",".join(questions_short).upper())
                    self.lsl_outlet.push_sample([",".join(answers)])
                    print("Marker sent: " + ",".join(answers))
                    writer.writerow(answers)
                else:
                    writer.writerow([q["question"] for q in self.questions])
                    writer.writerow([self.responses.get(i, "") for i in range(len(self.questions))])
            #QMessageBox.information(self, "Survey Submitted", "Thank you for completing the survey!")
            
            
            self.start_reactiontest_button.show()  # Show the button to start the arrow paradigm
            self.cnt_num_reactiontest = 1
            # self.title_label.setText("Arrow Paradigm Ready")
            self.question_label.hide()
            self.image_label.hide()
            #self.layout.removeWidget(self.input_widget)
            #self.input_widget.deleteLater()
            self.submit_button.hide()
            self.prev_button.hide()
            self.next_button.hide()
            if self.slider_label:
                self.slider_label.hide()
            if self.input_widget:
                self.input_widget.hide()
            self.break_questions_start = True
            self.break_questions_start = True

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save the survey results: {e}")

    def start_arrow_paradigm(self):
        """Start the arrow paradigm."""
        self.lsl_outlet.push_sample([self.config["Markers"]["start_trial"].upper()])
        print("Marker sent: " + self.config["Markers"]["start_trial"].upper())
        self.start_button.hide()
        #clear screen
        self.image_label.clear()
        self.image_label.hide()
        #hide the start button
        # pause for a few seconds before starting the arrow paradigm
        print("Starting arrow paradigm...")
        QTimer.singleShot( 5000, self.start_arrow_paradigm_run)
    
    def start_arrow_paradigm_run(self):
        """Run the arrow paradigm."""
        #self.stimuli_sequence = [random.choice(self.config["stimuli"]) for _ in range(self.num_trials)]
        # make array with each stimuli being evenly distributed num_trial times
        #self.stimuli_sequence = [stimulus for stimulus in self.stimuli_sequence for _ in range(self.num_repetitions)]
        self.stimuli_sequence = []
        for stimulus in self.config["stimuli"]:
            self.stimuli_sequence.extend([stimulus]* self.num_trials)
        print(self.stimuli_sequence)
        random.shuffle(self.stimuli_sequence)
        
        self.current_trial = 0
        self.run_next_stimulus()

    def run_next_stimulus(self):
        """Run the next stimulus in the sequence."""
        if self.current_trial < len(self.stimuli_sequence):
            stimulus = self.stimuli_sequence[self.current_trial]
            self.current_trial += 1

            if stimulus in self.images:
                pixmap = QPixmap(self.images[stimulus]).scaled(400, 400, Qt.KeepAspectRatio)
                self.image_label.setPixmap(pixmap)
                self.image_label.show()
                self.lsl_outlet.push_sample([self.config["Markers"][stimulus].upper()])
                print(f"Marker sent: {self.config['Markers'][stimulus].upper()}")
                self.current_stimulus = stimulus

            QTimer.singleShot(self.stimulus_duration, self.show_cross)
        else:
            #QMessageBox.information(self, "Info", "Arrow paradigm complete!")
            # start the survey again
            self.current_run += 1
            self.current_page = 0
            self.responses = {}
            self.load_page()
    


    def start_reaction_time(self):
        """Start the reaction time measurement."""
        self.start_reactiontest_button.hide()
        self.accept_reaction_button.hide()
        self.image_label.clear()
        
        self.reactiongame_started = True
        self.reaction_started = False
        self.setWindowTitle("Reaction Time Test")
        #self.setGeometry(100, 100, 600, 400)
        # Set up label
    
        self.background_color = QColor("red")
        self.central_widget.setStyleSheet("background-color: red; color: white; font-family: Arial;")
        self.image_label.clear()
        self.image_label.show()
        self.image_label.setStyleSheet("font-size: 48px; font-weight: bold; color: white;")
        self.image_label.setText("Wait until green!")
        self.click_button.setText("Wait...")
        self.click_button.show()

        
        # Start timer to turn green after random delay
        delay = random.randint(1500, 4000)  # ms
        QTimer.singleShot(delay, self.go_green)


    def go_green(self):
        self.start_time = time.perf_counter()
        self.image_label.setText("CLICK the button!")
        self.central_widget.setStyleSheet("background-color: green; color: white; font-family: Arial;")
        self.click_button.setText("CLICK!")
        self.reaction_started = True

    def save_reaction_time(self):
        self.accept_reaction_button.hide()
        self.image_label.clear()
        self.central_widget.setStyleSheet("background-color: black; color: white; font-family: Arial;")
        
        self.image_label.setText(f"Reaction time saved!\n Saved {self.cnt_num_reactiontest}/{self.num_repetitions}")
        self.cnt_num_reactiontest += 1
        self.image_label.setStyleSheet("font-size: 48px; font-weight: bold; color: white;")
        self.image_label.show()
        
        # save reaction time to file
        participant_id = self.part_number
        current_session = self.cur_session
        Pnum = f"sub-P{str(participant_id).zfill(3)}"
        Snum = f"ses-S{str(current_session).zfill(3)}"
        Rnum = f"run-{str(self.current_run).zfill(3)}"
        folder_path = os.path.join(self.cwdPath, "data", Pnum, Snum)
        os.makedirs(folder_path, exist_ok=True)
        save_file = os.path.join(folder_path, f"{Pnum}_{Snum}_task-arrow_reaction_{Rnum}.csv")
        print(f"Saving survey results to {save_file}"   )
        send_str = f"{self.cnt_num_reactiontest-1},{self.reaction_time}"
        self.lsl_outlet.push_sample([send_str])
        print("Sending string:", send_str)
        with open(save_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([self.cnt_num_reactiontest, self.reaction_time])
            
        if self.cnt_num_reactiontest < self.num_repetitions:
            # Show the button to start the next reaction test
            self.start_reactiontest_button.show()
            self.click_button.hide()
            self.image_label.hide()
        else:
            # Reaction test complete, hide the button
            self.start_reactiontest_button.hide()
            self.click_button.hide()
            self.image_label.setText("Reaction test complete!")
            self.image_label.show()
            self.lsl_outlet.push_sample([self.config["Markers"]["reactiontest_end"]])
            print("Marker sent: " + self.config["Markers"]["reactiontest_end"])
            # Reset for next run
            self.cnt_num_reactiontest = 1
            if self.current_run < self.num_max_runs:
                self.start_button.show()
                
                self.setWindowTitle("Arrow Paradigm Ready")
                self.image_label.setText("Ready for next run!")
                self.image_label.show()
            else:
                self.image_label.setText("All runs complete! Thank you for participating!")
                self.image_label.show()
                self.start_button.hide()
                self.start_reactiontest_button.hide()
                self.click_button.hide()
                self.accept_reaction_button.hide()
                # Optionally, close the application or reset
                # self.close()  # Uncomment to close the app after completion
    def reaction_click(self):
        current = time.perf_counter()
        self.reactiongame_started = False
        self.click_button.hide()
        self.image_label.clear()
        self.image_label.show()
        # check if the reaction test has started
        if not self.reaction_started:
            self.image_label.setText("Too early! Try again.")
            self.central_widget.setStyleSheet("background-color: grey; color: white; font-family: Arial;")
        else:
            self.central_widget.setStyleSheet("background-color: black; color: white; font-family: Arial;")
            self.reaction_time = int((current - self.start_time) * 1000)
            self.clicked = True
            self.image_label.setText(f"Your reaction time: {self.reaction_time} ms")
            self.accept_reaction_button.show()
            self.central_widget.setStyleSheet("background-color: #000000; color: white; font-family: Arial;")
        # Check if the reaction test is complete
        if self.cnt_num_reactiontest>= self.num_repetitions:
            self.image_label.setText(f"Reaction test complete!")
            
            self.start_reactiontest_button.hide()
            self.lsl_outlet.push_sample([self.config["Markers"]["reactiontest_end"]])
            print("Marker sent: " + self.config["Markers"]["reactiontest_end"])
        else:
            self.start_reactiontest_button.show()
            # LSL stream marker for reaction test start
            self.lsl_outlet.push_sample([self.config["Markers"]["reactiontest_start"]])
            print("Marker sent: " + self.config["Markers"]["reactiontest_start"])
                 

    def show_cross(self):
        """Show a cross for onset"""
        # show cross image
        self.image_label.clear()
        self.image_label.setPixmap(QPixmap(self.images["cross"]).scaled(100, 100, Qt.KeepAspectRatio))
        self.image_label.show()
        self.image_label.setStyleSheet("background-color: black;")
        
        # send marker for cross
        self.lsl_outlet.push_sample([self.config['Markers'][self.current_stimulus]+ self.config["Markers"]["cross"].upper()])
        print("Marker sent: " + self.config['Markers'][self.current_stimulus]+ self.config["Markers"]["cross"].upper())
        QTimer.singleShot(self.cross_duration, self.clear_screen)
        
    def clear_screen(self):
        """Clear the screen and prepare for the next trial."""
        self.image_label.clear()
        # draw white small circle in the middle of the screen
        self.image_label.setStyleSheet("font-size: 48px; font-weight: bold; color: white;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")
        self.image_label.show()
        if self.current_stimulus:
            self.lsl_outlet.push_sample([self.config["Markers"]["offset"].upper()])
            print("Marker sent:" + self.config["Markers"]["offset"].upper())
        else:
            print("Warning: current_stimulus is None, no marker sent.")
        QTimer.singleShot(self.black_screen_duration, self.run_next_stimulus)

            

def main():
    # Get the current working directory
    cwd = os.getcwd()
    cwdPath = os.path.join(cwd )
    # get character for current session and participant number
    part_number = input("Enter participant number: ")
    if not part_number:
        cur_session = 0
        print("Participant number is empty. THIS IS TEST MODE! RESTART if not intended. ")
    else:
        # check if part number already exists in folder data
        part_number = part_number.strip()
        # if not part_number.isdigit():
        #     print("Invalid participant number. Please enter a number.")
        #     return
        # part_number = int(part_number)
        # if part_number < 1 or part_number > 999:
        #     print("Participant number must be between 1 and 999.")
        #     return
        part_numberstr = f"sub-P{str(part_number).zfill(3)}"
        # get current session number
        cur_session = input("Enter current session number: ")   
        cur_session = cur_session.strip()
        if not cur_session.isdigit():
            print("Invalid session number. Please enter a number.")
            return
        cur_session = int(cur_session)
        if cur_session < 1 or cur_session > 999:
            print("Session number must be between 1 and 999.")
            return
        cur_sessionstr = f"ses-S{str(cur_session).zfill(3)}"
        # create the data folder if it does not exist
        data_folder = os.path.join(cwdPath, "data", part_numberstr, cur_sessionstr)
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
            print(f"Created data folder: {data_folder}")
        else:
            print(f"Data folder already exists (delete or enter new number): {data_folder}")
            exit(0)
        
    app = QApplication(sys.argv)
    window = SurveyApp(cwdPath, part_number, cur_session)
    
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
