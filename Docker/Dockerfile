FROM deepmi/fastsurfer:gpu-v2.2.0 AS runtime

# Copy venv and FastSurferCNN
COPY . /NeonateVINNA/

# Set all Pythonpath references already
ENV PYTHONPATH=/NeonateVINNA:$PYTHONPATH
    
# Set Workdir and entrypoint
#  the script entrypoint ensures that our conda env is active
WORKDIR "/NeonateVINNA"
USER nonroot
ENTRYPOINT ["/fastsurfer/Docker/entrypoint.sh", "bash", "./run_scripts/single_img_inference.sh", "--no_docker"]
CMD ["--help"]
