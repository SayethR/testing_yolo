# Documentación: Kria KV260 + Vitis AI (Windows / WSL2)

Configuración de trabajo: PC HP Victus (Windows) + tarjeta AMD/Xilinx Kria KV260, para compilar y correr modelos (objetivo final: YOLO, detección de personas) en el DPU de la tarjeta.

**Repositorio del proyecto:** https://github.com/SayethR/testing_yolo — contiene el script `run_yolov4_vart.py` y los cambios del proyecto. Ver "Preparación del entorno → Target" para cómo clonarlo en la Kria.

**Estructura de este documento:**
1. **Configuración inicial** — lo mínimo para dejar la Kria conectada y utilizable, separado entre Host (PC) y Target (Kria).
2. **Preparación del entorno** — herramientas de desarrollo (compilación de modelos, repo del proyecto), separadas igual entre Host y Target.
3. **Workflow** — el proceso completo de principio a fin, explicando el propósito de cada paso.

---

## 1. Configuración inicial

### Host (PC)

#### Internet Connection Sharing (ICS)
**Para qué es necesaria:** la Kria no tiene WiFi integrado y el router está lejos del lugar de trabajo. ICS convierte la PC en un "mini router": comparte el WiFi hacia el puerto Ethernet, dándole a la Kria IP automática (DHCP) e internet — es lo primero que hay que dejar andando para que la Kria sea alcanzable y tenga salida a internet.
1. Panel de Control → Centro de redes y recursos compartidos → Cambiar configuración del adaptador.
2. Clic derecho en el adaptador WiFi → Propiedades → pestaña "Uso compartido".
3. Marcar "Permitir que los usuarios de otras redes se conecten a través de la conexión a Internet de este equipo".
4. Seleccionar el adaptador "Ethernet" (el físico) en el menú desplegable.
5. Confirmar el aviso — el adaptador Ethernet de la PC queda fijo en `192.168.137.1/24`, con DHCP activo para `192.168.137.2 – 192.168.137.254`.

**Problema conocido:** el servicio de ICS a veces "se cae" internamente sin desmarcar la casilla visualmente (síntoma: la Kria responde por SSH/red local, pero no tiene salida real a internet — falla DNS y conexiones TCP salientes). Si pasa esto:
- Reiniciar el servicio: `services.msc` → buscar "Internet Connection Sharing (ICS)" → clic derecho → Reiniciar.
- O más rápido: desmarcar y volver a marcar la casilla del paso 3 (reseleccionando "Ethernet").
- Verificar que el WiFi de la PC sí tenga internet en ese momento — sin eso, ICS no tiene nada que compartir.

### Target (Kria)

#### Primer acceso a la tarjeta (consola serial UART)
Necesario la primera vez, antes de tener red configurada, o como respaldo si el SSH deja de responder. Es el método oficial de "primer contacto" documentado por Xilinx.

1. Conectar un cable USB-C (con transferencia de datos, no solo carga) del puerto **J4** de la placa portadora del KV260 a un puerto USB de la PC.
2. Si Windows no detecta el chip automáticamente, instalar el driver FTDI VCP desde ftdichip.com/drivers/vcp-drivers/.
3. En Administrador de dispositivos → Puertos (COM y LPT), identificar el puerto COM de número más bajo entre los dos que aparecen (en esta sesión: **COM8**).
4. Abrir PuTTY → Connection type: Serial → Serial line: `COM8` → Speed: `115200`.
5. En Connection → Serial, confirmar: Data bits = 8, Stop bits = 1, Parity = None, Flow control = None.
6. Clic en "Open" y presionar Enter para ver el prompt `kria login:`.
7. Usuario/contraseña por defecto: `ubuntu` / `ubuntu` (pide cambiar contraseña en el primer login).

#### IP fija de la Kria
Para no tener que buscar la IP asignada por DHCP cada vez que se enciende la tarjeta — queda siempre en la misma dirección dentro del rango que reparte ICS, conservando también el internet (el gateway sigue siendo la PC).
1. Conectarse a la Kria (por SSH o UART) e identificar el perfil de conexión:
   ```bash
   nmcli connection show
   ```
2. Fijar la IP dentro del perfil correspondiente (ej. "Wired connection 1"):
   ```bash
   sudo nmcli connection modify "Wired connection 1" \
     ipv4.addresses 192.168.137.50/24 \
     ipv4.gateway 192.168.137.1 \
     ipv4.dns 192.168.137.1 \
     ipv4.method manual
   sudo nmcli connection up "Wired connection 1"
   ```
3. Verificar:
   ```bash
   ip addr show eth0
   ```

#### Cargar el overlay del DPU
**No persiste entre reinicios** — hay que cargarlo cada vez que se enciende la tarjeta, antes de poder correr cualquier inferencia.
```bash
sudo xmutil loadapp kv260-smartcam
sudo xmutil listapps   # confirmar que "Active_slot" no sea -1
```
Si ya hay algo cargado en el slot y da error, primero: `sudo xmutil unloadapp`, y luego repetir el `loadapp`.

**Nota:** el script `run_yolov4_vart.py` (ver "Workflow") ya incluye una verificación automática de este paso — si detecta que el overlay no está cargado, ofrece cargarlo antes de continuar, evitando que el proceso se cuelgue.

#### Encender / apagar de forma segura
```bash
ssh ubuntu@192.168.137.50
```
Para apagar:
```bash
sudo shutdown -h now
```
Espera a que se corte la sesión SSH/UART antes de desconectar la alimentación física — apagar de golpe puede corromper la microSD. Para reiniciar en vez de apagar: `sudo reboot`.

---

## 2. Preparación del entorno

### Host (PC Windows / WSL2)

#### WSL2 con Ubuntu 20.04
**Para qué es necesaria:** Vitis AI (cuantización/compilación de modelos) no corre nativo en Windows — necesita un entorno Linux.
```powershell
wsl --install -d Ubuntu-20.04
```

#### NVIDIA Container Toolkit
**Para qué es necesaria:** permite que los contenedores Docker dentro de WSL accedan a la GPU NVIDIA de la PC, para acelerar la cuantización. *(Nota: en la práctica se terminó usando la imagen Docker CPU porque la build GPU local falla por un problema de licenciamiento de Anaconda — error 403 en `repo.anaconda.com`, issue conocido sin resolver en el repositorio oficial de Xilinx. El toolkit queda instalado por si se resuelve más adelante.)*
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo service docker restart
```
Verificar: `docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu20.04 nvidia-smi`

#### Docker Engine (dentro de WSL)
**Para qué es necesaria:** corre los contenedores oficiales de Vitis AI, que empaquetan todo el toolchain de cuantización/compilación.
```bash
sudo apt install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
sudo service docker start
```
Verificar: `docker run hello-world`

#### Repositorio e imagen Docker de Vitis-AI
**Para qué es necesaria:** contiene el código fuente y las imágenes Docker con el toolchain completo (cuantizador, compilador) para preparar modelos para el DPU del KV260.
```bash
cd ~
git clone --recurse-submodules https://github.com/Xilinx/Vitis-AI
cd Vitis-AI
docker pull xilinx/vitis-ai-cpu:2.5.0
```
**Importante:** usar específicamente la imagen `xilinx/vitis-ai-cpu:2.5.0` (versión combinada, todos los frameworks en un solo contenedor) — **no** `xilinx/vitis-ai-pytorch-cpu:latest`. La imagen `:latest` corresponde a Vitis AI 3.5, que genera `.xmodel` en un formato incompatible con el runtime `libvart 2.5.0` que trae la Kria (causó segfaults difíciles de diagnosticar — ver "Workflow" más abajo).

---

### Target (Kria)

#### Verificar los bindings de Python para VART
**Para qué es necesaria:** correr inferencia sin depender de `vitis-ai-library` (ver nota abajo). Ya vienen preinstalados en la imagen del sistema de la Kria — solo hay que confirmarlo.
```bash
python3 -c "import vart, xir, numpy, cv2"
```
Si todos importan sin error, no hace falta instalar nada más.

#### Nota sobre `vitis-ai-library` (paquete de `apt`) — evitarlo
El paquete `vitis-ai-library` instalable vía `apt` solo existe en versión **2.0**, y entra en conflicto con `libvart 2.5.0` (ya preinstalado en la Kria) — mezclar ambas versiones causa segfaults por incompatibilidad de ABI de C++, incluso corrigiendo manualmente los symlinks de las librerías compartidas. **No se recomienda instalarlo.** La solución adoptada fue un script en Python que usa `vart` y `xir` directamente (ver "Workflow", paso final).

#### Acceso SSH a GitHub (para clonar/actualizar el repositorio del proyecto)
**Para qué es necesaria:** GitHub ya no acepta usuario+contraseña para operaciones de git — hace falta una llave SSH asociada a la cuenta para clonar el repositorio del proyecto y mantenerlo sincronizado.
1. Generar la llave en la Kria:
   ```bash
   ssh-keygen -t ed25519 -C "kria-kv260"
   ```
   (Enter para aceptar la ubicación por defecto; se puede dejar la passphrase vacía)
2. Copiar la llave pública:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```
3. En GitHub (desde cualquier PC): perfil → Settings → SSH and GPG keys → New SSH key → pegar la llave copiada.
4. Clonar usando la URL SSH (no HTTPS):
   ```bash
   git clone git@github.com:SayethR/testing_yolo.git
   ```
   El repositorio incluye `run_yolov4_vart.py` — cualquiera con acceso a él puede replicar el script sin necesitar este documento aparte.

---

## 3. Workflow

Proceso completo, de principio a fin, para llevar un modelo (ej. YOLOv4) desde el Model Zoo hasta corriendo inferencia real sobre el DPU del KV260.

### Paso 1 — Confirmar el fingerprint exacto del DPU
**Propósito:** el DPU del KV260 es arquitectura **B3136**, pero eso no es una sola variante — hay parámetros internos que también deben coincidir exactamente, identificados por un "fingerprint". Compilar solo con el nombre genérico (`DPUCZDX8G_ISA1_B3136`) puede generar un modelo con fingerprint ligeramente distinto al real de la tarjeta y causar un segfault silencioso al cargar el modelo.
```bash
xdputil query
```
Anotar el valor de `"fingerprint"` (ej. `0x101000016010406`) del DPU activo — requiere haber cargado el overlay primero (ver "Configuración inicial → Target").

### Paso 2 — Descargar el modelo cuantizado
**Propósito:** obtener el modelo entrenado y ya cuantizado (no el paquete "precompilado para board", que puede venir compilado para otra variante de arquitectura, como pasó aquí con B4096 en vez de B3136).
```bash
wget https://www.xilinx.com/bin/public/openDownload?filename=<nombre_paquete>.zip -O modelo.zip
unzip modelo.zip
```
El archivo relevante queda en `quantized/quantize_eval_model.pb` (para modelos TensorFlow 1.x).

### Paso 3 — Crear el arch.json con el fingerprint exacto
**Propósito:** decirle al compilador exactamente qué variante de DPU tiene la tarjeta, sin ambigüedad.
```bash
echo '{"fingerprint": "0x101000016010406"}' > arch_exact.json
```

### Paso 4 — Compilar dentro del contenedor Docker correcto
**Propósito:** traducir el modelo cuantizado a instrucciones específicas para el DPU (el `.xmodel` final) — esto es análogo a compilar para una arquitectura de procesador específica, y debe hacerse con la versión de Vitis AI (2.5.0) que coincide con el runtime de la Kria.
```bash
cd ~/Vitis-AI
./docker_run.sh xilinx/vitis-ai-cpu:2.5.0
conda activate vitis-ai-tensorflow   # TensorFlow 1.x; usar vitis-ai-tensorflow2 o vitis-ai-pytorch según el framework del modelo
vai_c_tensorflow \
  --frozen_pb <ruta>/quantized/quantize_eval_model.pb \
  --arch arch_exact.json \
  --output_dir compile_output \
  --net_name <nombre_del_modelo>
```

### Paso 5 — Transferir el .xmodel a la Kria
```powershell
scp \\wsl.localhost\Ubuntu-20.04\home\<usuario>\Vitis-AI\compile_output\<nombre_del_modelo>.xmodel ubuntu@192.168.137.50:/home/ubuntu/
```

### Paso 6 — Colocar el modelo y verificar el fingerprint
**Propósito:** confirmar que el modelo recién compilado realmente coincide con el DPU físico antes de intentar correrlo — evita repetir el ciclo de debugging de segfaults.
```bash
sudo mkdir -p /usr/share/vitis_ai_library/models/<nombre_del_modelo>
sudo cp /home/ubuntu/<nombre_del_modelo>.xmodel /usr/share/vitis_ai_library/models/<nombre_del_modelo>/
xdputil xmodel /usr/share/vitis_ai_library/models/<nombre_del_modelo>/<nombre_del_modelo>.xmodel -l | grep fingerprint
```
Debe coincidir exactamente con el fingerprint del Paso 1.

### Paso 7 — Correr la inferencia (script Python, VART directo)
**Propósito:** ejecutar el modelo sobre el DPU y obtener las detecciones, evitando `vitis-ai-library` (ver nota en "Preparación del entorno → Target").

El script `run_yolov4_vart.py` (disponible en el repositorio del proyecto — ver "Preparación del entorno → Target"):
- Verifica automáticamente que el overlay del DPU esté cargado antes de continuar (y ofrece cargarlo si no lo está, evitando que el proceso se cuelgue).
- Carga el `.xmodel` con `xir`, crea un `runner` de `vart` (bajo nivel).
- Preprocesa la imagen: letterbox a 416×416 (mantiene proporción, rellena con gris) + normaliza.
- Cuantiza manualmente la entrada a enteros de 8 bits (usando el `fixpos` del tensor).
- Corre la inferencia, descuantiza las salidas.
- Decodifica las cajas de YOLO (sigmoid + anchors por escala) y aplica NMS.
- Imprime las detecciones y guarda `resultado_deteccion.jpg` con las cajas dibujadas.

```bash
python3 /home/ubuntu/testing_yolo/run_yolov4_vart.py \
  /usr/share/vitis_ai_library/models/yolov4_leaky_416_tf/yolov4_leaky_416_tf.xmodel \
  /home/ubuntu/test.jpg
```

**Resultado validado:** 5 detecciones sobre una foto de prueba (4 personas con confianza 0.88–0.96, 1 silla con 0.68) — DPU B3136 confirmado corriendo la inferencia real (861 operaciones del modelo mapeadas al subgrafo DPU), no el CPU ARM.

---

*Documento generado a partir de la sesión de configuración inicial del KV260. Modelo YOLOv4-leaky (COCO, 80 clases) compilado y validado corriendo detección de personas sobre el DPU B3136 vía script VART directo. Próximos pasos pendientes: adaptar a procesamiento de video en tiempo real, y profiling de rendimiento con vaitrace (Vitis AI Profiler).*
