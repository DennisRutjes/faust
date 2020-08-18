/************************************************************************
 ************************************************************************
    FAUST compiler
    Copyright (C) 2017 GRAME, Centre National de Creation Musicale
    ---------------------------------------------------------------------
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 ************************************************************************
 ************************************************************************/

#include "rust_code_container.hh"
#include "Text.hh"
#include "exception.hh"
#include "fir_function_builder.hh"
#include "floats.hh"
#include "global.hh"

#include <regex>
#include <sstream>

using namespace std;

/*
 Rust backend description:

 - 'usize' type has to be used for all array access: cast index as 'usize' only when using it (load/store arrays)
 - TODO: local stack variables (shared computation) are normally non-mutable
 - inputN/outputN local buffer variables in 'compute' are not created at all: they are replaced directly in the code
 with inputs[N]/outputs[N] (done in instructions_compiler.cpp)
 - BoolOpcode BinOps always casted to integer
 - 'delete' for SubContainers is not generated
 - add 'kMutable' and 'kReference' address access type

*/

map<string, bool> RustInstVisitor::gFunctionSymbolTable;

dsp_factory_base* RustCodeContainer::produceFactory()
{
    return new text_dsp_factory_aux(
        fKlassName, "", "",
        ((dynamic_cast<ostringstream*>(fOut)) ? dynamic_cast<ostringstream*>(fOut)->str() : ""), "");
}

CodeContainer* RustCodeContainer::createScalarContainer(const string& name, int sub_container_type)
{
    return new RustScalarCodeContainer(name, 0, 1, fOut, sub_container_type);
}

CodeContainer* RustCodeContainer::createContainer(const string& name, int numInputs, int numOutputs, ostream* dst)
{
    gGlobal->gDSPStruct = true;
    CodeContainer* container;

    if (gGlobal->gMemoryManager) {
        throw faustexception("ERROR : -mem not supported for Rust\n");
    }
    if (gGlobal->gFloatSize == 3) {
        throw faustexception("ERROR : quad format not supported for Rust\n");
    }
    if (gGlobal->gOpenCLSwitch) {
        throw faustexception("ERROR : OpenCL not supported for Rust\n");
    }
    if (gGlobal->gCUDASwitch) {
        throw faustexception("ERROR : CUDA not supported for Rust\n");
    }

    if (gGlobal->gOpenMPSwitch) {
        // container = new RustOpenMPCodeContainer(name, numInputs, numOutputs, dst);
        throw faustexception("ERROR : OpenMP not supported for Rust\n");
    } else if (gGlobal->gSchedulerSwitch) {
        // container = new RustWorkStealingCodeContainer(name, numInputs, numOutputs, dst);
        throw faustexception("ERROR : Scheduler not supported for Rust\n");
    } else if (gGlobal->gVectorSwitch) {
        // container = new RustVectorCodeContainer(name, numInputs, numOutputs, dst);
        throw faustexception("ERROR : Vector not supported for Rust\n");
    } else {
        container = new RustScalarCodeContainer(name, numInputs, numOutputs, dst, kInt);
    }

    return container;
}

void RustCodeContainer::produceInternal()
{
    int n = 0;

    // Global declarations
    tab(n, *fOut);
    fCodeProducer.Tab(n);
    generateGlobalDeclarations(&fCodeProducer);

    tab(n, *fOut);
    *fOut << "pub struct " << fKlassName << " {";

    tab(n + 1, *fOut);

    // Fields
    fCodeProducer.Tab(n + 1);
    generateDeclarations(&fCodeProducer);

    back(1, *fOut);
    *fOut << "}";

    tab(n, *fOut);
    tab(n, *fOut);
    *fOut << "impl " << fKlassName << " {";

    tab(n + 1, *fOut);
    tab(n + 1, *fOut);
    produceInfoFunctions(n + 1, fKlassName, "&self", false, false, &fCodeProducer);

    // Init
    // TODO
    // generateInstanceInitFun("instanceInit" + fKlassName, false, false)->accept(&fCodeProducer);

    tab(n + 1, *fOut);
    *fOut << "fn instance_init" << fKlassName << "(&mut self, sample_rate: i32) {";
    tab(n + 2, *fOut);
    fCodeProducer.Tab(n + 2);
    generateInit(&fCodeProducer);
    generateResetUserInterface(&fCodeProducer);
    generateClear(&fCodeProducer);
    back(1, *fOut);
    *fOut << "}";

    // Fill
    tab(n + 1, *fOut);
    string counter = "count";
    if (fSubContainerType == kInt) {
        tab(n + 1, *fOut);
        *fOut << "fn fill" << fKlassName << subst("(&mut self, $0: i32, table: &mut[i32]) {", counter);
    } else {
        tab(n + 1, *fOut);
        *fOut << "fn fill" << fKlassName << subst("(&mut self, $0: i32, table: &mut[$1]) {", counter, ifloat());
    }
    tab(n + 2, *fOut);
    fCodeProducer.Tab(n + 2);
    generateComputeBlock(&fCodeProducer);
    SimpleForLoopInst* loop = fCurLoop->generateSimpleScalarLoop(counter);
    loop->accept(&fCodeProducer);
    back(1, *fOut);
    *fOut << "}" << endl;

    tab(n, *fOut);
    *fOut << "}" << endl;

    // Memory methods
    tab(n, *fOut);
    tab(n, *fOut);
    *fOut << "pub fn new" << fKlassName << "() -> " << fKlassName << " { ";
    tab(n + 1, *fOut);
    *fOut << fKlassName << " {";
    RustInitFieldsVisitor initializer(fOut, n + 2);
    generateDeclarations(&initializer);
    tab(n + 1, *fOut);
    *fOut << "}";
    tab(n, *fOut);
    *fOut << "}";
}

void RustCodeContainer::generateWASMBuffers(int n) {
    // inout buffers
    for (int i = 0; i < fNumInputs; i++) {
        tab(n, *fOut);
        *fOut << "#[no_mangle]";
        tab(n, *fOut);
        *fOut << "static mut IN_BUFFER" << i << ": [f32;MAX_BUFFER_SIZE] = [0.;MAX_BUFFER_SIZE];";
    }

    // output buffers
    for (int i = 0; i < fNumOutputs; i++) {
        tab(n, *fOut);
        *fOut << "#[no_mangle]";
        tab(n, *fOut);
        *fOut << "static mut OUT_BUFFER" << i << ": [f32;MAX_BUFFER_SIZE] = [0.;MAX_BUFFER_SIZE];";
    }

    tab(n, *fOut);
    *fOut << "static mut INPUTS: [* const f32;" << fNumInputs 
          << "] = [0 as * const f32; " << fNumInputs << "];";

    tab(n, *fOut);
    *fOut << "static mut OUTPUTS: [* mut f32;" << fNumOutputs 
          << "] = [0 as * mut f32; " << fNumOutputs << "];";
}

void RustCodeContainer::produceClass()
{
    int n = 0;

    // Sub containers
    generateSubContainers();

    tab(n, *fOut);
    fCodeProducer.Tab(n);
    generateGlobalDeclarations(&fCodeProducer);
    
    // generate global audio buffers 
    generateWASMBuffers(n);

    // determine the number of required voices
    int nVoices = calculateNumVoices();

    // static Buffer for FAUST DSP instance
    tab(n, *fOut);
    *fOut << "static mut ENGINE " << ": " << fKlassName << " = " << fKlassName << " {";
    RustInitFieldsVisitor initializer1(fOut, n + 1);
    generateDeclarations(&initializer1);
    if (nVoices > 0) {
        generateVoicesDeclarationInit(n, nVoices);
    }
    tab(n, *fOut);
    *fOut << "};" << "\n\n";

    *fOut << "type T = " << ifloat() << ";\n";

    tab(n, *fOut);

    *fOut << "struct " << fKlassName << " {";
    tab(n + 1, *fOut);

    // Fields
    fCodeProducer.Tab(n + 1);
    generateDeclarations(&fCodeProducer);
    back(1, *fOut);
    if (nVoices > 0) {
        generateVoicesDeclarations(n, nVoices);
    }
    tab(n, *fOut);
    *fOut << "}";
    tab(n, *fOut);

    tab(n, *fOut);
    *fOut << "impl " << fKlassName << " {";

    // Associated type
    //tab(n + 1, *fOut);
    //*fOut << "type T = " << ifloat() << ";";

    // Memory methods
    tab(n + 2, *fOut);
    if (fAllocateInstructions->fCode.size() > 0) {
        tab(n + 2, *fOut);
        *fOut << "static void allocate" << fKlassName << "(" << fKlassName << "* dsp) {";
        tab(n + 2, *fOut);
        fAllocateInstructions->accept(&fCodeProducer);
        back(1, *fOut);
        *fOut << "}";
    }

    tab(n + 1, *fOut);

    if (fDestroyInstructions->fCode.size() > 0) {
        tab(n + 1, *fOut);
        *fOut << "static void destroy" << fKlassName << "(" << fKlassName << "* dsp) {";
        tab(n + 2, *fOut);
        fDestroyInstructions->accept(&fCodeProducer);
         back(1, *fOut);
        *fOut << "}";
        tab(n + 1, *fOut);
    }

    *fOut << "fn new() -> " << fKlassName << " { ";
    if (fAllocateInstructions->fCode.size() > 0) {
        tab(n + 2, *fOut);
        *fOut << "allocate" << fKlassName << "(dsp);";
    }
    tab(n + 2, *fOut);
    *fOut << fKlassName << " {";
    RustInitFieldsVisitor initializer(fOut, n + 3);
    generateDeclarations(&initializer);
    if (nVoices > 0) {
        generateVoicesDeclarationInit(n+3, nVoices);
    }
    tab(n + 2, *fOut);
    *fOut << "}";
    tab(n + 1, *fOut);
    *fOut << "}";

    // Print metadata declaration
    //produceMetadata(n + 1);
    
    // this will determine the number of voices, if greater than 0, otherwise default is 0
    produceVoices(n + 1, nVoices);

    produceSetGetBuffers(n + 1);

    // Get sample rate method
    tab(n + 1, *fOut);
    fCodeProducer.Tab(n + 1);
    *fOut << "pub " ;
    generateGetSampleRate("get_sample_rate", "&self", false, false)->accept(&fCodeProducer);

    produceInfoFunctions(n + 1, "", "&self", false, false, &fCodeProducer);

    // Inits

    // TODO
    //
    // CInstVisitor codeproducer1(fOut, "");
    // codeproducer1.Tab(n+2);
    // generateStaticInitFun("classInit" + fKlassName, false)->accept(&codeproducer1);
    // generateInstanceInitFun("instanceInit" + fKlassName, false, false)->accept(&codeproducer2);

    tab(n + 1, *fOut);
    *fOut << "fn class_init(sample_rate: i32) {";
    {
        tab(n + 2, *fOut);
        // Local visitor here to avoid DSP object type wrong generation
        RustInstVisitor codeproducer(fOut, "");
        codeproducer.Tab(n + 2);
        generateStaticInit(&codeproducer);
    }
    back(1, *fOut);
    *fOut << "}";

    tab(n + 1, *fOut);
    *fOut << "fn instance_reset_params(&mut self) {";
    {
        tab(n + 2, *fOut);
        // Local visitor here to avoid DSP object type wrong generation
        RustInstVisitor codeproducer(fOut, "");
        codeproducer.Tab(n + 2);
        generateResetUserInterface(&codeproducer);
    }
    back(1, *fOut);
    *fOut << "}";

    tab(n + 1, *fOut);
    *fOut << "fn instance_clear(&mut self) {";
    {
        tab(n + 2, *fOut);
        // Local visitor here to avoid DSP object type wrong generation
        RustInstVisitor codeproducer(fOut, "");
        codeproducer.Tab(n + 2);
        generateClear(&codeproducer);
    }
    back(1, *fOut);
    *fOut << "}";

    tab(n + 1, *fOut);
    *fOut << "fn instance_constants(&mut self, sample_rate: i32) {";
    {
        tab(n + 2, *fOut);
        // Local visitor here to avoid DSP object type wrong generation
        RustInstVisitor codeproducer(fOut, "");
        codeproducer.Tab(n + 2);
        generateInit(&codeproducer);
    }
    back(1, *fOut);
    *fOut << "}";

    tab(n + 1, *fOut);
    *fOut << "fn instance_init(&mut self, sample_rate: i32) {";
    tab(n + 2, *fOut);
    *fOut << "self.instance_constants(sample_rate);";
    tab(n + 2, *fOut);
    *fOut << "self.instance_reset_params();";
    tab(n + 2, *fOut);
    *fOut << "self.instance_clear();";
    tab(n + 1, *fOut);
    *fOut << "}";

    tab(n + 1, *fOut);
    *fOut << "pub fn init(&mut self, sample_rate: i32) {";
    tab(n + 2, *fOut);
    *fOut << fKlassName << "::class_init(sample_rate);";
    tab(n + 2, *fOut);
    *fOut << "self.instance_init(sample_rate);";
    tab(n + 2, *fOut);
    *fOut << "self.init_voices();";
    tab(n + 2, *fOut);
    *fOut << "self.init_buffers();";
    tab(n + 1, *fOut);
    *fOut << "}";

    // Pre-pass of user interface instructions to determine parameter lookup table (field name => index)
    UserInterfaceParameterMapping parameterMappingVisitor;
    fUserInterfaceInstructions->accept(&parameterMappingVisitor);
    auto parameterLookup = parameterMappingVisitor.getParameterLookup();

    // User interface (non-static method)
    // tab(n + 1, *fOut);
    // tab(n + 1, *fOut);
    // *fOut << "fn build_user_interface(&self, ui_interface: &mut dyn UI<Self::T>) {";
    // tab(n + 2, *fOut);
    // *fOut << "Self::build_user_interface_static(ui_interface);";
    // tab(n + 1, *fOut);
    // *fOut << "}";

    // User interface (static method)
    tab(n + 1, *fOut);
    *fOut << "pub fn get_param_info(&mut self, name: &str) -> Param {";
    tab(n + 2, *fOut);
    *fOut << "match name {";
    tab(n + 3, *fOut);
    fCodeProducer.Tab(n + 3);
    RustUIInstVisitor uiCodeproducer(fOut, "", parameterLookup, n + 3);
    generateUserInterface(&uiCodeproducer);
    *fOut << "_ => Param { index: -1, range: ParamRange::new(0.0, 0.0, 0.0, 0.0)}";
    tab(n + 2, *fOut);
    *fOut << "}";
    tab(n + 1, *fOut);
    *fOut << "}";

    // init voices
    initVoices(n+1, nVoices);
    handleNoteEvent(n+1, nVoices);

    initBuffers(n+1);
    

    // Parameter getter/setter
    produceParameterGetterSetter(n + 1, parameterLookup);

    // Compute
    generateCompute(n + 1);

    // Compute external
    generateComputeExternal(n + 1);

    tab(n, *fOut);
    *fOut << "}" << endl;
    tab(n, *fOut);
}

void RustCodeContainer::handleNoteEvent(int n, int nVoices) 
{
    tab(n, *fOut);

    if (nVoices > 1) {
        *fOut << "pub fn handle_note_on(&mut self, mn: Note, vel: f32) {";
        tab(n+1, *fOut);
        *fOut << "let mut allocated_voice = 0;";
        tab(n+1, *fOut);
        *fOut << "let mut allocated_voice_age = self.voices[allocated_voice].voice_age;";
        tab(n+1, *fOut);
        *fOut << "// find the oldest voice to reuse";
        tab(n+1, *fOut);
        *fOut << "for i in 0.." << nVoices << " {";
        tab(n+2, *fOut);
        *fOut << "let age = self.voices[i].voice_age;";
        tab(n+2, *fOut);
        *fOut << "if age < allocated_voice_age {";
        tab(n+3, *fOut);
        *fOut << "allocated_voice_age = age;";
        tab(n+3, *fOut);
        *fOut << "allocated_voice = i;";
        tab(n+2, *fOut);
        *fOut << "}";
        tab(n+1, *fOut);
        *fOut << "}";

        tab(n+1, *fOut);
        *fOut << "// update the VoiceInfo for our chosen voice";
        tab(n+1, *fOut);
        *fOut << "self.voices[allocated_voice].channel   = 0;";
        tab(n+1, *fOut);
        *fOut << "self.voices[allocated_voice].note      = mn;";
        tab(n+1, *fOut);
        *fOut << "self.voices[allocated_voice].voice_age = self.next_allocated_voice_age;";
        tab(n+1, *fOut);
		*fOut << "self.next_allocated_voice_age          = self.next_allocated_voice_age + 1;";
        *fOut << "// set params for choosen voice";
		tab(n+1, *fOut);
        *fOut << "self.set_param(self.voice_gate[allocated_voice], 1.0);";
        tab(n+1, *fOut);
        *fOut << "self.set_param(self.voice_gain[allocated_voice], vel);";
        tab(n+1, *fOut);
        *fOut << "self.set_param(self.voice_freq[allocated_voice], to_freq(mn));";
        tab(n, *fOut);
        *fOut << "}" << endl;
        
        tab(n, *fOut);
        *fOut << "pub fn handle_note_off(&mut self, mn: Note, vel: f32) {";						
        tab(n+1, *fOut);
        *fOut << "for voice in 0.." << nVoices << " {";
        tab(n+2, *fOut);
        *fOut << "if self.voices[voice].note == mn {";
        tab(n+3, *fOut);
        *fOut << "// mark voice as being unused";
        tab(n+3, *fOut);
		*fOut << "self.voices[voice].voice_age = self.next_unallocated_voice_age;";
        tab(n+3, *fOut);
		*fOut << "self.next_unallocated_voice_age = self.next_unallocated_voice_age + 1;";
        tab(n+3, *fOut);
        *fOut << "// set params for choosen voice";
        tab(n+3, *fOut);
        *fOut << "self.set_param(self.voice_gate[voice], 0.0);";
        tab(n+3, *fOut);
        *fOut << "self.set_param(self.voice_gain[voice], vel);";
        tab(n+2, *fOut);
        *fOut << "}";

        tab(n+1, *fOut);
        *fOut << "}";

        tab(n, *fOut);
        *fOut << "}" << endl;
        
    }
    else if (nVoices == 1) {
        *fOut << "pub fn handle_note_on(&mut self, mn: Note, vel: f32) {";
        tab(n+1, *fOut);
        *fOut << "// set params for voice";
		tab(n+1, *fOut);
        *fOut << "self.set_param(self.voice_gate[0], 1.0);";
        tab(n+1, *fOut);
        *fOut << "self.set_param(self.voice_gain[0], vel);";
        tab(n+1, *fOut);
        *fOut << "self.set_param(self.voice_freq[0], to_freq(mn));";
        tab(n, *fOut);
        *fOut << "}";
        
        tab(n, *fOut);
        *fOut << "pub fn handle_note_off(&mut self, mn: Note, vel: f32) {";						
        tab(n+1, *fOut);
        *fOut << "// set params for voice";
        tab(n+1, *fOut);
        *fOut << "self.set_param(self.voice_gate[0], 0.0);";
        tab(n+1, *fOut);
        *fOut << "self.set_param(self.voice_gain[0], vel);";
        tab(n+2, *fOut);
        tab(n, *fOut);
        *fOut << "}";
        
    }
    else {
        *fOut << "pub fn handle_note_on(&mut self, _mn: Note, _vel: f32) {";
        tab(n, *fOut);
        *fOut << "}";

        *fOut << "pub fn handle_note_off(&mut self, _mn: Note, _vel: f32) {";
        tab(n, *fOut);
        *fOut << "}";
    }
    


    

}

void RustCodeContainer::initVoices(int n, int nVoices) 
{
    tab(n, *fOut);
    *fOut << "fn init_voices(&mut self) {";
    for (int i=0; i < nVoices; i++) {
        tab(n+1, *fOut);
        *fOut << "self.voice_freq[" << i << "] = self.get_param_info(\"freq_v" << i << "\").index as u32;";
        tab(n+1, *fOut);
        *fOut << "self.voice_gain[" << i << "] = self.get_param_info(\"gain_v" << i << "\").index as u32;";
        tab(n+1, *fOut);
        *fOut << "self.voice_gate[" << i << "] = self.get_param_info(\"gate_v" << i << "\").index as u32;";
    }
    tab(n, *fOut);
    *fOut << "}";
}

void RustCodeContainer::initBuffers(int n) 
{
    tab(n, *fOut);
    *fOut << "fn init_buffers(&self) {";
    
    tab(n+1, *fOut);
    *fOut << "unsafe {" ;

    for (int i=0; i < fNumInputs; i++) {
        tab(n+2, *fOut);
        *fOut << "INPUTS[" << i << "] = IN_BUFFER" << i << ".as_ptr();";
    }
    for (int i=0; i < fNumOutputs; i++) {
        tab(n+2, *fOut);
        *fOut << "OUTPUTS[" << i << "] = OUT_BUFFER" << i << ".as_mut_ptr();";
    }

    tab(n+1, *fOut);
    *fOut << "};" ;

    tab(n, *fOut);
    *fOut << "}";
}

void RustCodeContainer::generateVoicesDeclarations(int n, int nVoices) 
{
    tab(n+1, *fOut);
    *fOut << "next_allocated_voice_age: i64,";
    tab(n+1, *fOut);
	*fOut << "next_unallocated_voice_age: i64,";
    tab(n+1, *fOut);
	*fOut << "voices: [VoiceInfo;" << nVoices << "],";
    tab(n+1, *fOut);
	*fOut << "voice_freq: [u32;" << nVoices << "],";
    tab(n+1, *fOut);
	*fOut << "voice_gain: [u32;" << nVoices << "],";
    tab(n+1, *fOut);
	*fOut << "voice_gate: [u32;" << nVoices << "],";
}

void RustCodeContainer::generateVoicesDeclarationInit(int n, int nVoices) 
{
    tab(n+1, *fOut);
    *fOut << "next_allocated_voice_age: 1000000000,";
    tab(n+1, *fOut);
	*fOut << "next_unallocated_voice_age: 0,";
    tab(n+1, *fOut);
	*fOut << "voices: [VoiceInfo {active: false,note: 0,channel: 0,voice_age: 0,};" << nVoices << "],";
    tab(n+1, *fOut);
	*fOut << "voice_freq: [0;" << nVoices << "],";
    tab(n+1, *fOut);
	*fOut << "voice_gain: [0;" << nVoices << "],";
    tab(n+1, *fOut);
	*fOut << "voice_gate: [0;" << nVoices << "],";
}

enum STR2INT_ERROR { S_SUCCESS, S_OVERFLOW, S_UNDERFLOW, S_INCONVERTIBLE };

STR2INT_ERROR str2int (int &i, char const *s, int base = 0)
{
    char *end;
    long  l;
    errno = 0;
    l = strtol(s, &end, base);
    if ((errno == ERANGE && l == LONG_MAX) || l > INT_MAX) {
        return S_OVERFLOW;
    }
    if ((errno == ERANGE && l == LONG_MIN) || l < INT_MIN) {
        return S_UNDERFLOW;
    }
    if (*s == '\0' || *end != '\0') {
        return S_INCONVERTIBLE;
    }
    i = l;
    return S_SUCCESS;
}

int RustCodeContainer::calculateNumVoices()
{
    for (auto& i : gGlobal->gMetaDataSet) {
        if (i.first == tree("aavoices")) {
            stringstream my_stream(ios::in|ios::out);
            my_stream << **(i.second.begin());
            string str(my_stream.str());
            str.erase(remove( str.begin(), str.end(), '\"' ),str.end());
            int nVoices = 0;
            str2int(nVoices, str.c_str(), 10);
            return nVoices;
        } 
    }
    return 0;
}

void RustCodeContainer::produceVoices(int n, int nVoices)
{
    tab(n, *fOut);
    *fOut << "pub fn get_voices(&self) -> i32 { ";
    tab(n + 1, *fOut);
    *fOut << nVoices;
    tab(n, *fOut);
    *fOut << "}" << endl;
}

void RustCodeContainer::produceSetGetBuffers(int n) 
{
    tab(n, *fOut);
    *fOut << "pub fn get_input(&self, index: u32) -> u32 { ";
    tab(n + 1, *fOut);
    *fOut << "unsafe { INPUTS[index as usize] as u32 }";
    tab(n, *fOut);
    *fOut << "}" << endl;

    tab(n, *fOut);
    *fOut << "pub fn get_output(&self, index: u32) -> u32 { ";
    tab(n + 1, *fOut);
    *fOut << "unsafe { OUTPUTS[index as usize] as u32 }";
    tab(n, *fOut);
    *fOut << "}" << endl;

    tab(n, *fOut);
    *fOut << "pub fn set_input(&self, index: u32, offset: u32) { ";
    tab(n + 1, *fOut);
    *fOut << "unsafe { INPUTS[index as usize] = offset as * const f32; };";
    tab(n, *fOut);
    *fOut << "}" << endl;

    tab(n, *fOut);
    *fOut << "pub fn set_output(&self, index: u32, offset: u32) { ";
    tab(n + 1, *fOut);
    *fOut << "unsafe { OUTPUTS[index as usize] = offset as * mut f32; };";
    tab(n, *fOut);
    *fOut << "}" << endl;
}

void RustCodeContainer::produceMetadata(int n)
{
    tab(n, *fOut);
    *fOut << "fn metadata(&self, m: &mut dyn Meta) { ";

    // We do not want to accumulate metadata from all hierachical levels, so the upper level only is kept
    for (auto& i : gGlobal->gMetaDataSet) {
        if (i.first != tree("author")) {
            tab(n + 1, *fOut);
            *fOut << "m.declare(\"" << *(i.first) << "\", " << **(i.second.begin()) << ");";
        } else {
            // But the "author" meta data is accumulated, the upper level becomes the main author and sub-levels become
            // "contributor"
            for (set<Tree>::iterator j = i.second.begin(); j != i.second.end(); j++) {
                if (j == i.second.begin()) {
                    tab(n + 1, *fOut);
                    *fOut << "m.declare(\"" << *(i.first) << "\", " << **j << ");";
                } else {
                    tab(n + 1, *fOut);
                    *fOut << "m.declare(\""
                          << "contributor"
                          << "\", " << **j << ");";
                }
            }
        }
    }

    tab(n, *fOut);
    *fOut << "}" << endl;
}

void RustCodeContainer::produceInfoFunctions(int tabs, const string& classname, const string& obj, bool ismethod, bool isvirtual,
                                             TextInstVisitor* producer)
{
    producer->Tab(tabs);
    *fOut << "pub " ;
    generateGetInputs(subst("get_num_inputs$0", classname), obj, false, false)->accept(&fCodeProducer);
    *fOut << "pub " ;
    generateGetOutputs(subst("get_num_outputs$0", classname), obj, false, false)->accept(&fCodeProducer);
    producer->Tab(tabs);
    *fOut << "pub " ;
    generateGetInputRate(subst("get_input_rate$0", classname), obj, false, false)->accept(&fCodeProducer);
    producer->Tab(tabs);
    *fOut << "pub " ;
    generateGetOutputRate(subst("get_output_rate$0", classname), obj, false, false)->accept(&fCodeProducer);
}

void RustCodeContainer::produceParameterGetterSetter(int tabs, map<string, int> parameterLookup)
{
    // Add `get_param`
    tab(tabs, *fOut);
    tab(tabs, *fOut);
    *fOut << "pub fn get_param(&self, param: u32) -> T {";
    tab(tabs + 1, *fOut);
    *fOut << "match param {";
    for (const auto &paramPair : parameterLookup) {
        const auto fieldName = paramPair.first;
        const auto index = paramPair.second;
        tab(tabs + 2, *fOut);
        *fOut << index << " => self." << fieldName << ",";
    }
    tab(tabs + 2, *fOut);
    *fOut << "_ => 0.,";
    tab(tabs + 1, *fOut);
    *fOut << "}";
    tab(tabs, *fOut);
    *fOut << "}";

    // Add `set_param`
    tab(tabs, *fOut);
    tab(tabs, *fOut);
    *fOut << "pub fn set_param(&mut self, param: u32, value: T) {";
    tab(tabs + 1, *fOut);
    *fOut << "match param {";
    for (const auto &paramPair : parameterLookup) {
        const auto fieldName = paramPair.first;
        const auto index = paramPair.second;
        tab(tabs + 2, *fOut);
        *fOut << index << " => { self." << fieldName << " = value }";
    }
    tab(tabs + 2, *fOut);
    *fOut << "_ => {}";
    tab(tabs + 1, *fOut);
    *fOut << "}";
    tab(tabs, *fOut);
    *fOut << "}";
}

// Scalar
RustScalarCodeContainer::RustScalarCodeContainer(const string& name, int numInputs, int numOutputs, std::ostream* out,
                                                 int sub_container_type)
    : RustCodeContainer(name, numInputs, numOutputs, out)
{
    fSubContainerType = sub_container_type;
}

void RustScalarCodeContainer::generateComputeExternal(int n) {
    tab(n, *fOut);
    *fOut << "#[inline]";
    tab(n, *fOut);
    *fOut << "pub fn compute_external(&mut self, count: i32) {";
    tab(n+1, *fOut);
    *fOut << "let (";
    for (int i = 0; i < fNumInputs; i++) {
        *fOut << "input" << i << ", ";
    }
    for (int i = 0; i < fNumOutputs; i++) {
        *fOut << "output" << i;
        if (i + 1 != fNumOutputs) {
            *fOut << ", ";
        }
    }
    *fOut << ") = unsafe {";
    tab(n+2, *fOut);
    *fOut << "(";
    for (int i = 0; i < fNumInputs; i++) {
        *fOut << "::std::slice::from_raw_parts(INPUTS[" << i << "], count as usize),";
        //*fOut << "INPUTS[" << i << "],";
        tab(n+2, *fOut);
    }
    for (int i = 0; i < fNumOutputs; i++) {
        //*fOut << "OUTPUTS[" << i << "]";
        *fOut << "::std::slice::from_raw_parts_mut(OUTPUTS[" << i << "], count as usize)";
        if (i + 1 != fNumOutputs) {
            *fOut << ",";
            tab(n+2, *fOut);
        }
    }
    *fOut << ")";
    tab(n+1, *fOut);
    *fOut << "};";
    tab(n+1, *fOut);
    *fOut << "unsafe { self.compute(count, &[";
    for (int i = 0; i < fNumInputs; i++) {
        *fOut << "input" << i;
        if (i + 1 != fNumInputs) {
            *fOut << ", ";
        }
    }
    *fOut << "], &mut [";
    for (int i = 0; i < fNumOutputs; i++) {
        *fOut << "output" << i;
        if (i + 1 != fNumOutputs) {
            *fOut << ", ";
        }
    }
    *fOut << "]); }";

    tab(n, *fOut);
    *fOut << "}";
}

void RustScalarCodeContainer::generateCompute(int n)
{
    // Generates declaration
    tab(n, *fOut);
    // add WASM simd stuff so we can get it to auto vectorize
    *fOut << "#[target_feature(enable = \"simd128\")]";
    tab(n, *fOut);
    *fOut << "#[inline]";

    tab(n, *fOut);
    *fOut << "unsafe fn compute(&mut self, " << fFullCount << ": i32, ";
    if (fNumInputs == 0) {
        *fOut << "inputs: &[T], ";
    }
    else {
        *fOut << "inputs: &[&[T];" << fNumInputs << "], ";
    }

    *fOut << "outputs: &mut [&mut [T];" << fNumOutputs << "]) {";

    tab(n + 1, *fOut);
    fCodeProducer.Tab(n + 1);

    // Generates local variables declaration and setup
    generateComputeBlock(&fCodeProducer);

    // Generates one single scalar loop
    std::vector<std::string> iterators;
    for (int i = 0; i < fNumInputs; ++i) {
        iterators.push_back("inputs" + std::to_string(i));
    }
    for (int i = 0; i < fNumOutputs; ++i) {
        iterators.push_back("outputs"+ std::to_string(i));
    }
    IteratorForLoopInst* loop = fCurLoop->generateSimpleScalarLoop(iterators);
    loop->accept(&fCodeProducer);

    back(1, *fOut);
    *fOut << "}" << endl;
}

// Vector
RustVectorCodeContainer::RustVectorCodeContainer(const string& name, int numInputs, int numOutputs, std::ostream* out)
    : VectorCodeContainer(numInputs, numOutputs), RustCodeContainer(name, numInputs, numOutputs, out)
{
}

void RustVectorCodeContainer::generateComputeExternal(int n) {
    
}


void RustVectorCodeContainer::generateCompute(int n)
{
    // Possibly generate separated functions
    fCodeProducer.Tab(n);
    tab(n, *fOut);
    generateComputeFunctions(&fCodeProducer);

    // Compute declaration
    tab(n, *fOut);
    *fOut << "fn compute("
          << subst("&mut self, $0: i32, inputs: &[&[Self::T]], outputs: &mut[&mut[Self::T]]) {", fFullCount);
    tab(n + 1, *fOut);
    fCodeProducer.Tab(n + 1);

    // Generates local variables declaration and setup
    generateComputeBlock(&fCodeProducer);

    // Generates the DSP loop
    fDAGBlock->accept(&fCodeProducer);

    back(1, *fOut);
    *fOut << "}" << endl;
}

// OpenMP
RustOpenMPCodeContainer::RustOpenMPCodeContainer(const string& name, int numInputs, int numOutputs, std::ostream* out)
    : OpenMPCodeContainer(numInputs, numOutputs), RustCodeContainer(name, numInputs, numOutputs, out)
{
}

void RustOpenMPCodeContainer::generateComputeExternal(int n) {
    
}


void RustOpenMPCodeContainer::generateCompute(int n)
{
    // Possibly generate separated functions
    fCodeProducer.Tab(n);
    tab(n, *fOut);
    generateComputeFunctions(&fCodeProducer);

    // Compute declaration
    tab(n, *fOut);
    *fOut << "fn compute("
          << subst("&mut self, $0: i32, inputs: &[&[Self::T]], outputs: &mut[&mut[Self::T]]) {", fFullCount);
    tab(n + 1, *fOut);
    fCodeProducer.Tab(n + 1);

    // Generates local variables declaration and setup
    generateComputeBlock(&fCodeProducer);

    // Generate it
    fGlobalLoopBlock->accept(&fCodeProducer);

    back(1, *fOut);
    *fOut << "}" << endl;
}

// Works stealing scheduler
RustWorkStealingCodeContainer::RustWorkStealingCodeContainer(const string& name, int numInputs, int numOutputs,
                                                             std::ostream* out)
    : WSSCodeContainer(numInputs, numOutputs, "dsp"), RustCodeContainer(name, numInputs, numOutputs, out)
{
}

void RustWorkStealingCodeContainer::generateComputeExternal(int n) {
    
}


void RustWorkStealingCodeContainer::generateCompute(int n)
{
    // Possibly generate separated functions
    fCodeProducer.Tab(n);
    tab(n, *fOut);
    generateComputeFunctions(&fCodeProducer);

    // Generates "computeThread" code
    // Note that users either have to adjust the trait in their architecture file.
    // Alternatively we would have to attach this method to the impl, not the trait.
    tab(n, *fOut);
    *fOut << "pub fn compute_thread(" << fKlassName << "&mut self, num_thread: i32) {";
    tab(n + 1, *fOut);
    fCodeProducer.Tab(n + 1);

    // Generate it
    fThreadLoopBlock->accept(&fCodeProducer);

    tab(n, *fOut);
    *fOut << "}" << endl;

    // Compute "compute" declaration
    tab(n, *fOut);
    *fOut << "fn compute("
          << subst("&mut self, $0: i32, inputs: &[&[Self::T]], outputs: &mut[&mut[Self::T]]) {", fFullCount);
    tab(n + 1, *fOut);
    fCodeProducer.Tab(n + 1);

    // Generates local variables declaration and setup
    generateComputeBlock(&fCodeProducer);

    tab(n, *fOut);
    *fOut << "}" << endl;

    tab(n, *fOut);
    *fOut << "extern \"C\" void computeThreadExternal(&mut self, num_thread: i32) {";
    tab(n + 1, *fOut);
    *fOut << "compute_thread((" << fKlassName << "*)dsp, num_thread);";
    tab(n, *fOut);
    *fOut << "}" << endl;
}
