DEPDIR := .d
$(shell mkdir -p $(DEPDIR) >/dev/null)

COMPILE_DOE   := python3 ../../compile_doe.py
CREATE_SCRIPT := python3 ../../create_script.py
CREATE_MAP    := python3 ../../create_treatment_map.py

MAIN_FILE:=main.run.sh
DEST_DIR=$(patsubst %/$(MAIN_FILE),%,$@)

DESIGNS  := $(wildcard *.design)
INCLUDES := $(wildcard *.include.sh)
COMPILED := $(patsubst %.design,%.yaml, $(DESIGNS))
DEPS     := $(patsubst %.design,$(DEPDIR)/%.d, $(DESIGNS))

.PHONY: all run test map clean clean-script clean-run clean-select deps

all: run noseg map

# ===== This is an automated recipe for making run-scripts with selection: no-segmention =====

NOSEG_SUFFIX:=.noseg.select.dir
NOSEG_LIST:=$(patsubst %.design,%$(NOSEG_SUFFIX)/$(MAIN_FILE), $(DESIGNS))
NOSEG_OPTIONS:=--select 'f=-1'

noseg: $(NOSEG_LIST)

$(NOSEG_LIST): $(INCLUDES)

%$(NOSEG_SUFFIX)/$(MAIN_FILE): %.yaml
	@rm -f $(DEST_DIR)/*.run.sh || true
	@rmdir $(DEST_DIR) &>/dev/null || true
	$(CREATE_SCRIPT) $< -o $(DEST_DIR) $(NOSEG_OPTIONS)


# ===== This is an automated recipe for making run-scripts without special options =====

RUN_SUFFIX:=.run.dir
RUN_LIST:=$(patsubst %.design,%$(RUN_SUFFIX)/$(MAIN_FILE), $(DESIGNS))

run: $(RUN_LIST)

$(RUN_LIST): $(INCLUDES)

%$(RUN_SUFFIX)/$(MAIN_FILE): %.yaml
	@rm -f $(DEST_DIR)/*.run.sh || true
	@rmdir $(DEST_DIR) &>/dev/null || true
	$(CREATE_SCRIPT) $< -o $(DEST_DIR)

# ===== Test scripts --- a bit different because files are created/not dirs =====

TEST_SUFFIX:=.test.sh
TEST_LIST:=$(patsubst %.design,%$(TEST_SUFFIX), $(DESIGNS))
TEST_OPTIONS:=--test -1

test: $(TEST_LIST)

$(TEST_LIST): $(INCLUDES)

%$(TEST_SUFFIX): %.yaml
	$(CREATE_SCRIPT) $< -o $@ $(TEST_OPTIONS)

# ===== This is an automated recipe for compiling the .design files =====

%.yaml: %.design $(DEPDIR)/%.d
	$(COMPILE_DOE) $< -o $@

# ===== This is an automated recipe for creating the .map files =====

MAP_SUFFIX:=.map
MAP_LIST:=$(patsubst %.design,%$(MAP_SUFFIX), $(DESIGNS))

map: $(MAP_LIST)

%$(MAP_SUFFIX): %.yaml
	$(CREATE_MAP) $< -o $@

# ===== Those are recipes to clean-up the build =====

clean: clean-script
	-rm -f *.yaml *.test.sh $(DEPDIR)/*.d
	-rmdir -p $(DEPDIR)

clean-script: clean-run clean-select

clean-select:
	-for f in *.select.dir; do rm -f "$$f"/*.run.sh; done || true
	-rmdir *.select.dir

clean-run:
	-for f in *.run.dir; do rm -f "$$f"/*.run.sh; done || true
	-rmdir *.run.dir

# ===== Those are recipes to keep auto-dependencies =====

# TODO: This is still not fully automated

deps: $(DEPS)

$(DEPDIR)/%.d : %.design
	$(COMPILE_DOE) $< -o $(patsubst %.design, %.yaml, $<) -d $@
	@# $(COMPILE_DOE) $< -o $(patsubst %.design, %.yaml, $<) -d $(patsubst %.d,%.Td,$@)
	@# sed 's/\:/ $(DEPDIR)\/$(notdir $@) \:/' < $(patsubst %.d,%.Td,$@) > $@
	@# rm $(patsubst %.d,%.Td,$@)
.PRECIOUS: $(DEPDIR)/%.d

include $(wildcard $(DEPDIR)/*.d)
#include $(wildcard $(patsubst %,$(DEPDIR)/%.d,$(DESIGNS)))
