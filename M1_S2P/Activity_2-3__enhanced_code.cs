// Define the pin number for the push button and LED
const uint8_t BTN_PIN = 2;
const uint8_t LED_PIN = 13;

// Store the previous button state and current LED state
uint8_t buttonPrevState = LOW;
uint8_t ledState = LOW;

// Configure the board before the main program starts
void setup()
{
    // Set the button pin as an input using the internal pull-up resistor
    pinMode(BTN_PIN, INPUT_PULLUP);

    // Set the LED pin as an output
    pinMode(LED_PIN, OUTPUT);

    // Start serial communication for monitoring values
    Serial.begin(9600);
}

void loop()
{
    // Read the current state of the push button
    uint8_t buttonState = digitalRead(BTN_PIN);

    // Print the current button state, previous button state, and LED state
    Serial.print(buttonState);
    Serial.print(buttonPrevState);
    Serial.print(ledState);
    Serial.println("");

    // Toggle LED only when button is newly pressed (LOW to HIGH)
    // This prevents double-toggle on press+release cycle
    if (buttonState == HIGH && buttonPrevState == LOW)
    {
        ledState = !ledState;
        digitalWrite(LED_PIN, ledState);
    }

    // Save current state for next loop iteration
    buttonPrevState = buttonState;

    // Short delay for basic debouncing, improved responsiveness
    delay(50);
}