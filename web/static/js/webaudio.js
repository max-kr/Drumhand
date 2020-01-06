
const audioBufferSourceNodeContext = new AudioContext()
const drumKitSoundNames = [
    'hihat',
    'clap',
    'bass'
];

const drumKitBuffers = [];
    // loop through the sounds we want to import
    for(let soundName of drumKitSoundNames) {
        // fetch them from the file system
        fetch('/sounds/' + soundName + '.mp3')
            // when we get the asynchronous response, convert to an ArrayBuffer
            .then(response => response.arrayBuffer())
            .then(buffer => {
                // decode the ArrayBuffer as an AudioBuffer
                audioBufferSourceNodeContext.decodeAudioData(buffer, decoded => {
                    // push the resulting sound to an array
                    drumKitBuffers.push(decoded);
                });
            });
    }

let audioBufferSourceNode;

const playDrums = (index) => {
    // allow the user to play sound
    audioBufferSourceNodeContext.resume();
    if(audioBufferSourceNode) audioBufferSourceNode.stop();

    // create a new AudioBufferSourceNode
    audioBufferSourceNode = audioBufferSourceNodeContext.createBufferSource();

    // set the buffer to the appropriate index
    audioBufferSourceNode.buffer = drumKitBuffers[index];

    // connect the buffer node to the destination
    audioBufferSourceNode.connect(audioBufferSourceNodeContext.destination);

    // start playing the sound
    audioBufferSourceNode.start();
}

