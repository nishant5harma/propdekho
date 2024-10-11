import os
from pyannote.audio import Pipeline
import speech_recognition as sr
from pydub import AudioSegment

def enhance_audio(input_file, output_file):
    """
    Perform basic noise reduction and export enhanced audio.
    """
    audio = AudioSegment.from_file(input_file)
    
    # Normalize the audio to a reasonable level
    normalized_audio = audio.normalize()
    
    # Apply a simple noise reduction (e.g., filtering low-level noise)
    reduced_noise_audio = normalized_audio.low_pass_filter(3000)
    
    # Export to .wav for further processing
    reduced_noise_audio.export(output_file, format="wav")

def diarize_audio(audio_file):
    """Perform speaker diarization using pyannote.audio"""
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0",use_auth_token="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")
    diarization = pipeline(audio_file)
    
    speaker_segments = []
    for segment, track, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append((segment.start, segment.end, speaker))

    return speaker_segments

def transcribe_segment(recognizer, audio_file, start_time, duration):
    """Transcribe a specific segment of audio."""
    with sr.AudioFile(audio_file) as source:
        try:
            audio = recognizer.record(source, offset=start_time, duration=duration)
            transcription = recognizer.recognize_google(audio)
            return transcription
        except sr.UnknownValueError:
            return "[Unintelligible]"
        except sr.RequestError:
            return "[Error in transcription]"

def assign_speakers_and_transcribe(speaker_segments, audio_file):
    """Merge diarization and transcription results."""
    recognizer = sr.Recognizer()
    conversation = []
    
    for start, end, speaker in speaker_segments:
        duration = end - start
        
        # Transcribe this segment
        transcription = transcribe_segment(recognizer, audio_file, start, duration)
        
        # Append the transcription with speaker label
        conversation.append(f"Person {speaker}: {transcription}")
    
    return conversation

def write_to_txt(conversation, output_file="call_transcription.txt"):
    """Write the conversation to a text file."""
    with open(output_file, "w") as f:
        for line in conversation:
            f.write(line + "\n")

def process_audio_call(audio_file):
    print("Enhancing the audio...")
    enhanced_audio_file = "enhanced_audio.wav"
    enhance_audio(audio_file, enhanced_audio_file)
    
    print("Performing speaker diarization...")
    speaker_segments = diarize_audio(enhanced_audio_file)
    
    print("Assigning speakers and transcribing the audio...")
    conversation = assign_speakers_and_transcribe(speaker_segments, enhanced_audio_file)
    
    print("Writing transcription to file...")
    write_to_txt(conversation)

# Example usage
audio_file_path = "voice.aac"  # Convert this to .wav first
process_audio_call(audio_file_path)
