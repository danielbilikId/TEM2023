function write_serial_TEM(num)
item = uint8(num); % options to choose:  1,2,3
divide_factor = uint8(1); %max value = 255 min value = 1
try 
fclose(instrfind); % instrfind = returns all valid serial port objects as an array to out
end

port = 'COM3';  % teensy Port
uart = serial(port, 'BaudRate', 2000);
fopen(uart);

pause(3);
fwrite(uart,item); % send bytes
fwrite(uart,divide_factor); % send bytes
fclose(instrfind);
end 