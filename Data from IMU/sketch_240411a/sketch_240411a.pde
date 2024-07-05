String[] lines; // Array to store lines from the file
int index = 0; // Index to keep track of current line
int totalLines; // Total number of lines in the file

void setup() {
  size(800, 600, P3D);
  lines = loadStrings("imu_data.txt"); // Change "imu_data.txt" to your file name
  totalLines = lines.length;
}

void draw() {
  background(233);
  translate(width / 2, height / 2, 0);
  
  if (index < totalLines) {
    // Parse the current line to get pitch and roll
    String line = lines[index];
    float roll = extractValue(line, "Roll:");
    float pitch = extractValue(line, "Pitch:");
    
    // Display pitch and roll
    textSize(22);
    text("Roll: " + int(roll) + "     Pitch: " + int(pitch), -100, 265);
    
    // Rotate the object
    rotateX(radians(-pitch));
    rotateZ(radians(roll));
    
    // Draw the box
    fill(0, 76, 153);
    box(100, 20, 50); // Adjust size as needed
    
    // Move to the next line for the next frame
    index++;
  } else {
    // If reached the end of the file, stop the animation
    noLoop();
  }
}

float extractValue(String line, String keyword) {
  int start = line.indexOf(keyword) + keyword.length();
  int end = line.indexOf(" ", start);
  return float(line.substring(start, end));
}
