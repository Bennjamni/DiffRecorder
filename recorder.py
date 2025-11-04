import sys
import time
from datetime import datetime

try:
	import sounddevice as sd
	import soundfile as sf
except Exception as e:
	print("Erro ao importar dependências: ", e)
	print("Instale as dependências com: pip install sounddevice soundfile")
	sys.exit(1)


def list_input_devices():
	devices = sd.query_devices()
	inputs = []
	print("Dispositivos de entrada disponíveis:")
	for i, dev in enumerate(devices):
		if dev.get('max_input_channels', 0) > 0:
			inputs.append(i)
			print(f"  {i}: {dev['name']}  (max_input_channels={dev['max_input_channels']})")
	if not inputs:
		print("Nenhum dispositivo de entrada encontrado.")
	return inputs


def choose_device():
	inputs = list_input_devices()
	if not inputs:
		sys.exit(1)
	while True:
		try:
			choice = input("Escolha o índice do dispositivo (ex: 0): ").strip()
			idx = int(choice)
			dev = sd.query_devices(idx)
			if dev.get('max_input_channels', 0) <= 0:
				print("O dispositivo selecionado não tem canais de entrada. Escolha outro.")
				continue
			return idx, dev
		except ValueError:
			print("Por favor digite um número válido.")
		except Exception as e:
			print("Erro ao selecionar dispositivo:", e)


def choose_samplerate():
	print("Escolha a taxa de amostragem (sempre em 24 bits):")
	print("  1) 48 kHz (24-bit)")
	print("  2) 192 kHz (24-bit)")
	while True:
		c = input("Opção (1 ou 2): ").strip()
		if c == '1':
			return 48000
		if c == '2':
			return 192000
		print("Escolha 1 ou 2.")


def ask_channels(default_channels):
	prompt = f"Número de canais (pressione Enter para usar {default_channels}): "
	while True:
		c = input(prompt).strip()
		if c == '':
			return default_channels
		try:
			nc = int(c)
			if nc <= 0:
				print("Número de canais deve ser > 0")
				continue
			return nc
		except ValueError:
			print("Digite um número inteiro para canais.")


def ask_duration():
	print("Duração em segundos (digite um número) ou pressione Enter para gravar até Ctrl+C")
	while True:
		c = input("Duração (s) [Enter = grava até Ctrl+C]: ").strip()
		if c == '':
			return None
		try:
			val = float(c)
			if val <= 0:
				print("Digite um valor positivo.")
				continue
			return val
		except ValueError:
			print("Digite um número válido (por exemplo 5 ou 2.5).")


def record_to_file(device, samplerate, channels, duration):
	timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
	filename = f"recording_{timestamp}_{samplerate}Hz.wav"
	subtype = 'PCM_24'

	print(f"Arquivo de saída: {filename}")
	print(f"Dispositivo: {device}, taxa: {samplerate} Hz, canais: {channels}")

	try:
		if duration is None:
			# grava até Ctrl+C
			with sf.SoundFile(filename, mode='w', samplerate=samplerate,
							  channels=channels, subtype=subtype) as file:
				with sd.InputStream(samplerate=samplerate, device=device,
									 channels=channels, dtype='float32') as stream:
					print("Gravando... pressione Ctrl+C para parar.")
					while True:
						data, _ = stream.read(1024)
						file.write(data)
		else:
			frames = int(duration * samplerate)
			print(f"Gravando por {duration} segundos ({frames} frames)...")
			rec = sd.rec(frames=frames, samplerate=samplerate, channels=channels,
						 dtype='float32', device=device)
			sd.wait()
			sf.write(filename, rec, samplerate, subtype=subtype)

		print("Gravação finalizada com sucesso.")
	except KeyboardInterrupt:
		print("\nGravação interrompida pelo usuário (Ctrl+C). Salvando arquivo...")
		# se estivermos usando SoundFile no contexto, o arquivo já terá sido fechado corretamente
		print("Arquivo salvo (verifique o diretório atual).")
	except Exception as e:
		print("Erro durante gravação:", e)


def main():
	print("=== Gravador de Áudio (linha de comando) ===")
	device_idx, dev = choose_device()
	samplerate = choose_samplerate()
	default_channels = min(dev.get('max_input_channels', 1), 2)
	channels = ask_channels(default_channels)
	duration = ask_duration()

	# Teste rápido de abertura de stream para verificar compatibilidade de taxa/canais
	try:
		with sd.InputStream(samplerate=samplerate, device=device_idx, channels=channels):
			pass
	except Exception as e:
		print("Atenção: não foi possível abrir o dispositivo com as configurações escolhidas:", e)
		print("Você pode tentar outra combinação de dispositivo/canais/taxa.")
		if not input("Deseja tentar novamente? (s/N): ").strip().lower().startswith('s'):
			sys.exit(1)
		return main()

	record_to_file(device_idx, samplerate, channels, duration)


if __name__ == '__main__':
	main()

