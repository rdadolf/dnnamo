DEVICE_NAMES=\
  tf_cpu

build:
	@( for b in $(DEVICE_NAMES); do $(MAKE) DEVICE_NAME="$$b" -C $$b/ubench; done )

clean:
	@( for b in $(DEVICE_NAMES); do $(MAKE) DEVICE_NAME="$$b" -C $$b/ubench clean; done )
