"""Speech-to-text support for LiteLLM."""

from collections.abc import AsyncIterable
import base64
import io
from typing import override
import wave

from openai import OpenAIError
from websockets.exceptions import WebSocketException

from homeassistant.components import stt
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_API_KEY, CONF_MODEL, CONF_URL
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import LOGGER, STT_BATCH_ENDPOINT, STT_REALTIME_ENDPOINT
from .coordinator import LiteLLMConfigEntry, async_get_model_groups
from .entity import LiteLLMEntity


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: LiteLLMConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up LiteLLM STT entities."""
    stt_subentries = [
        subentry
        for subentry in config_entry.subentries.values()
        if subentry.subentry_type == "stt"
    ]
    if not stt_subentries:
        return

    model_groups = await async_get_model_groups(
        hass, config_entry.data[CONF_URL], config_entry.data.get(CONF_API_KEY)
    )
    endpoints_by_model = {
        model["model_group"]: model.get("supported_endpoints") or []
        for model in model_groups
        if model.get("mode") == "audio_transcription"
    }

    for subentry in stt_subentries:
        async_add_entities(
            [
                LiteLLMSTTEntity(
                    config_entry,
                    subentry,
                    endpoints_by_model.get(subentry.data[CONF_MODEL], []),
                )
            ],
            config_subentry_id=subentry.subentry_id,
        )


class LiteLLMSTTEntity(stt.SpeechToTextEntity, LiteLLMEntity):
    """LiteLLM speech-to-text entity."""

    def __init__(
        self,
        entry: LiteLLMConfigEntry,
        subentry: ConfigSubentry,
        supported_endpoints: list[str],
    ) -> None:
        """Initialize the STT entity."""
        super().__init__(entry, subentry)
        self._supported_endpoints = supported_endpoints

    @property
    @override
    def supported_languages(self) -> list[str]:
        """Return supported languages."""
        return [self.entry.runtime_data.hass.config.language]

    @property
    @override
    def supported_formats(self) -> list[stt.AudioFormats]:
        """Return supported formats."""
        return [stt.AudioFormats.WAV]

    @property
    @override
    def supported_codecs(self) -> list[stt.AudioCodecs]:
        """Return supported codecs."""
        return [stt.AudioCodecs.PCM]

    @property
    @override
    def supported_bit_rates(self) -> list[stt.AudioBitRates]:
        """Return supported bit rates."""
        return [stt.AudioBitRates.BITRATE_16]

    @property
    @override
    def supported_sample_rates(self) -> list[stt.AudioSampleRates]:
        """Return supported sample rates."""
        return [stt.AudioSampleRates.SAMPLERATE_16000]

    @property
    @override
    def supported_channels(self) -> list[stt.AudioChannels]:
        """Return supported channels."""
        return [stt.AudioChannels.CHANNEL_MONO]

    @override
    async def async_process_audio_stream(
        self, metadata: stt.SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> stt.SpeechResult:
        """Process an audio stream."""
        if STT_REALTIME_ENDPOINT in self._supported_endpoints:
            return await self._async_process_realtime(metadata, stream)
        if STT_BATCH_ENDPOINT in self._supported_endpoints:
            return await self._async_process_batch(metadata, stream)
        return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

    async def _async_process_batch(
        self, metadata: stt.SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> stt.SpeechResult:
        """Process audio with the transcription endpoint."""
        audio_bytes = bytearray()
        async for chunk in stream:
            audio_bytes.extend(chunk)

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(metadata.channel.value)
            wav_file.setsampwidth(metadata.bit_rate.value // 8)
            wav_file.setframerate(metadata.sample_rate.value)
            wav_file.writeframes(audio_bytes)

        try:
            response = await self.entry.runtime_data.client.with_options(
                max_retries=0
            ).audio.transcriptions.create(
                model=self.model,
                file=("audio.wav", wav_buffer.getvalue()),
                response_format="json",
                language=metadata.language.split("-")[0],
            )
        except OpenAIError:
            LOGGER.exception("Error during STT")
        else:
            if response.text:
                return stt.SpeechResult(
                    response.text, stt.SpeechResultState.SUCCESS
                )

        return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

    async def _async_process_realtime(
        self, metadata: stt.SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> stt.SpeechResult:
        """Stream audio to LiteLLM's realtime transcription endpoint."""
        try:
            async with self.entry.runtime_data.client.realtime.connect(
                model=self.model, extra_query={"intent": "transcription"}
            ) as connection:
                await connection.send(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "transcription",
                            "audio": {
                                "input": {
                                    "format": {
                                        "type": "audio/pcm",
                                        "rate": 16000,
                                        "channels": 1,
                                    },
                                    "transcription": {
                                        "model": self.model,
                                        "language": metadata.language.split("-")[0],
                                    },
                                }
                            },
                        },
                    }
                )
                async for chunk in stream:
                    await connection.input_audio_buffer.append(
                        audio=base64.b64encode(chunk).decode()
                    )
                await connection.input_audio_buffer.commit()

                async for event in connection:
                    if (
                        event.type
                        == "conversation.item.input_audio_transcription.completed"
                    ):
                        if event.transcript:
                            return stt.SpeechResult(
                                event.transcript, stt.SpeechResultState.SUCCESS
                            )
                        break
                    if event.type == "error":
                        break
        except (OpenAIError, WebSocketException, OSError):
            LOGGER.exception("Error during realtime STT")

        return stt.SpeechResult(None, stt.SpeechResultState.ERROR)
