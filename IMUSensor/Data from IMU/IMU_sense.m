% Define serial port parameters
port = '/dev/cu.usbmodem83301'; % Replace 'COMX' with the appropriate port name
baudrate = 9600; % Baud rate

% Open serial port
s = serialport(port, baudrate);

try
    while true
        % Read data from serial port
        data = readline(s);

        % Split data based on space delimiter
        parts = strsplit(data, ' ');

        % Parse data based on prefix (A for accelerometer, G for gyroscope)
        if strcmp(parts{1}, 'A')
            Ax = str2double(parts{2});
            Ay = str2double(parts{3});
            Az = str2double(parts{4});
            
            % Calculate pitch and roll angles
            pitch = atan2d(-Ax, sqrt(Ay^2 + Az^2));
            roll = atan2d(Ay, Az);
            
            fprintf('Pitch=%.2f degrees, Roll=%.2f degrees\n', pitch, roll);
        elseif strcmp(parts{1}, 'G')
            Gx = str2double(parts{2});
            Gy = str2double(parts{3});
            Gz = str2double(parts{4});
            
            % Assume small time step for simplicity
            dt = 0.1; % Change as needed
            
            % Integrate gyroscope data to obtain yaw
            yaw = integrateGyro(Gz, dt); % Implement this function
            
            fprintf('Yaw=%.2f degrees\n', yaw);
        else
            fprintf('Unknown data: %s\n', data);
        end
    end
catch ME
    disp(ME);
    % Close serial port
    clear s;
end
