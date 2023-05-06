function DSOX1204G()
% Find a VISA-USB object.
obj1 = instrfind('Type', 'visa-usb', 'RsrcName', 'USB0::0x2A8D::0x0396::CN59297151::0::INSTR', 'Tag', '');
% obj1 = instrfind('Type', 'visa-usb', 'RsrcName', 'USB0::0x2A8D::0x0396::CN59207300::0::INSTR', 'Tag', ''); 
% Create the VISA-USB object if it does not exist
% USB0::0x2A8D::0x0396::CN59207300::0::INSTR
% otherwise use the object that was found.

if isempty(obj1)
    obj1 = visa('KEYSIGHT', 'USB0::0x2A8D::0x0396::CN59297151::0::INSTR');
else
    fclose(obj1);
    obj1 = obj1(1);
end

% Configure instrument object, obj1.
set(obj1, 'InputBufferSize', 2000000);
set(obj1, 'OutputBufferSize', 1024);

% Configure instrument object, obj1.
set(obj1, 'ByteOrder', 'bigEndian');

fopen(obj1);
% Communicating with instrument object, obj1.
figure(1);
hold off
tiledlayout(2,2)
t_array = [];
data_array = [];

fprintf(obj1, 'WAVeform:POINts 2000000');
fprintf(obj1, 'WAVeform:POINts:MODE MAX');

for i=1:4
    
    nexttile
    fprintf(obj1, 'WAVEFORM:SOURCE CHAN' + string(i));
    fprintf(obj1, 'WAVEFORM:FORMAT WORD');

    fprintf(obj1, 'WAVEFORM:DATA?');
    data = binblockread(obj1, 'uint16');

    fprintf(obj1, 'WAVEFORM:PREAMBLE?');
    preamble = fscanf(obj1, '%g, %g, %g, %g, %g, %g, %g, %g, %g, %g');
    % <preamble_block> ::= <format 16-bit NR1>,
    % <type 16-bit NR1>,
    % <points 32-bit NR1>, preamble(3)
    % <count 32-bit NR1>,
    % <xincrement 64-bit floating point NR3>, preamble(5)
    % <xorigin 64-bit floating point NR3>,
    % <xreference 32-bit NR1>,
    % <yincrement 32-bit floating point NR3>,
    % <yorigin 32-bit floating point NR3>,
    % <yreference 32-bit NR1>

    % Prepare time axis
    t = [1:preamble(3)*2]*preamble(5)/2; % time in seconds
    t_array = [t_array t'];
    % prepare the y axis
    data = (data-preamble(10))*preamble(8) + preamble(9); % in volts
    data_array =[data_array data];
    
    %plot(t, data);
    %plot(data);
    %title('Channel ' + string(i));
end

%%
t1 = t_array(:,1);
y1 = data_array(:,1);
t2 = t_array(:,2);
y2 = data_array(:,2);
t3 = t_array(:,3);
y3 = data_array(:,3);
t4 = t_array(:,4);
y4 = data_array(:,4);

% save the data
save 'TEM_PYTHON/DSOX1204G_test.mat' t1 y1 t2 y2 t3 y3 t4 y4;

%% Disconnect and Clean Up
% Disconnect from instrument object, obj1.
fclose(obj1);

end