
class DPUOp:
    def __init__(self, name, symbol, type, kid):
        self.name = name
        self.symbol = symbol
        self.type = type
        self.kid = kid

class DPUBinaryOp(DPUOp):    
    def generate_macro(self, out):
        out.write(f"DEFINE_BINARY_KERNEL({self.type}, {self.name}, {self.symbol})\n")
    def kernel_name(self):
        return f"binary_{self.type}_{self.name}"
    @staticmethod
    def make(name, symbol):
        return lambda type, kid: DPUBinaryOp(name, symbol, type, kid)

class DPUBinaryScalarOp(DPUOp):    
    def generate_macro(self, out):
        out.write(f"DEFINE_BINARY_SCALAR_KERNEL({self.type}, {self.name}, {self.symbol})\n")
    def kernel_name(self):
        return f"binary_scalar_{self.type}_{self.name}"
    @staticmethod
    def make(name, symbol):
        return lambda type, kid: DPUBinaryScalarOp(name, symbol, type, kid)


class DPUUnaryOp(DPUOp):
    def generate_macro(self, out):
        if self.name == 'universal_pipeline':
            out.write(f"DEFINE_UNIVERSAL_PIPELINE_KERNEL({self.type})\n")
        else:
            out.write(f"DEFINE_UNARY_KERNEL({self.type}, {self.name}, {self.symbol})\n")
    def kernel_name(self):
        if self.name == 'universal_pipeline':
             return f"universal_{self.type}_pipeline"
        else:
             return f"unary_{self.type}_{self.name}"
    @staticmethod

    def make(name, symbol):
        return lambda type, kid: DPUUnaryOp(name, symbol, type, kid)
    

class DPUReduceOp(DPUOp):    
    def generate_macro(self, out):
        out.write(f"DEFINE_REDUCTION_KERNEL({self.type}, {self.name}, {self.symbol})\n")
    def kernel_name(self):
        return f"reduction_{self.type}_{self.name}"
    @staticmethod
    def make(name, symbol):
        return lambda type, kid: DPUReduceOp(name, symbol, type, kid)
    


def declare_unary_op(name, symbol):
    return lambda type, kid: DPUUnaryOp(name, symbol, type, kid)

categories = ['Binary', 'Unary', 'Reduction', 'BinaryScalar']

ops = [
    DPUBinaryOp.make('add', '+'),
    DPUBinaryOp.make('sub', '-'),
    DPUBinaryOp.make('mul', '*'),
    DPUBinaryOp.make('div', '/'),
    DPUBinaryOp.make('asr', '>>'),


    DPUUnaryOp.make('negate', 'NEGATE'),
    DPUUnaryOp.make('abs', 'ABS'),

    DPUReduceOp.make('min', 'MIN'),
    DPUReduceOp.make('max', 'MAX'),
    DPUReduceOp.make('sum', 'SUM'),
    DPUReduceOp.make('product', 'PRODUCT'),
]

types = [
    'int32_t'
]



# Experimental Universal Pipeline
universal_pipeline_ops = [
    DPUUnaryOp.make('universal_pipeline', 'universal_pipeline')
]

all_ops = [] # :: (KernelID, OpClass)
grouped_ops = []
kernel_id = 0
for type in types:
    group = []
    
    # Binary Ops
    for op in ops:
        if isinstance(op('int32_t', 0), DPUBinaryOp):
             op_instance = op(type, kernel_id)
             if op_instance.name == 'asr' and type != 'int32_t':
                 continue
             all_ops.append((kernel_id, op_instance))
             group.append(op_instance)
             kernel_id += 1
    
    # Binary Scalar Ops
    for op in ops:
        if isinstance(op('int32_t', 0), DPUBinaryOp):
             # Create a scalar version of the binary op
             base_op = op('int32_t', 0)
             if base_op.name == 'asr' and type != 'int32_t':
                 continue
             op_instance = DPUBinaryScalarOp(base_op.name + "_scalar", base_op.symbol, type, kernel_id)
             all_ops.append((kernel_id, op_instance))
             group.append(op_instance)
             kernel_id += 1
    
    # Individual Unary Ops (Restored)
    for op in ops:
        if isinstance(op('int32_t', 0), DPUUnaryOp):
             op_instance = op(type, kernel_id)
             all_ops.append((kernel_id, op_instance))
             group.append(op_instance)
             kernel_id += 1

    # Universal Pipeline Op (Experimental)
    for op in universal_pipeline_ops:
         op_instance = op(type, kernel_id)
         all_ops.append((kernel_id, op_instance))
         group.append(op_instance)
         kernel_id += 1


    # Reduction Ops
    for op in ops:
        if isinstance(op('int32_t', 0), DPUReduceOp):
             op_instance = op(type, kernel_id)
             all_ops.append((kernel_id, op_instance))
             group.append(op_instance)
             kernel_id += 1
             
    grouped_ops.append((type, group))


# Generate OpCodes include file
pipeline_ops = [
    ('identity', 'IDENTITY'),
    ('negate', 'NEGATE'),
    ('abs', 'ABS'),
     # Binary
    ('add', 'ADD'),
    ('sub', 'SUB'),
    ('mul', 'MUL'),
    ('div', 'DIV'),
    ('asr', 'ASR'),
    # Binary Scalar
    ('add_scalar', 'ADD_SCALAR'),
    ('sub_scalar', 'SUB_SCALAR'),
    ('mul_scalar', 'MUL_SCALAR'),
    ('div_scalar', 'DIV_SCALAR'),
    ('asr_scalar', 'ASR_SCALAR'),
    # Reduction 
    ('min', 'MIN'),
    ('max', 'MAX'),
    ('sum', 'SUM'),
    ('product', 'PRODUCT'),
    # Stack Machine
    ('push_input', 'PUSH_INPUT'),
    ('push_operand_0', 'PUSH_OPERAND_0'),
    ('push_operand_1', 'PUSH_OPERAND_1'),
    ('push_operand_2', 'PUSH_OPERAND_2'),
    ('push_operand_3', 'PUSH_OPERAND_3'),
    ('push_operand_4', 'PUSH_OPERAND_4'),
    ('push_operand_5', 'PUSH_OPERAND_5'),
    ('push_operand_6', 'PUSH_OPERAND_6'),
    ('push_operand_7', 'PUSH_OPERAND_7'),
]

with open("common/opcodes.h", "w") as out:
    out.write('#ifndef OPCODES_H\n')
    out.write('#define OPCODES_H\n\n')
    out.write('// GENERATED BY tools/generate_kernels.py. DO NOT EDIT\n\n')
    out.write('enum OpCode {\n')
    for idx, (name, symbol) in enumerate(pipeline_ops):
        out.write(f'    OP_{symbol} = {idx},\n')
    out.write('};\n\n')
    
    # Generate classification macros
    out.write('#define IS_OP_STACK(op) ((op) >= OP_PUSH_INPUT && (op) <= OP_PUSH_OPERAND_7)\n')
    out.write('#define IS_OP_UNARY(op) ((op) >= OP_NEGATE && (op) <= OP_ABS)\n')
    out.write('#define IS_OP_BINARY(op) ((op) >= OP_ADD && (op) <= OP_ASR)\n')
    out.write('#define IS_OP_SCALAR(op) ((op) >= OP_ADD_SCALAR && (op) <= OP_ASR_SCALAR)\n')
    out.write('#define IS_OP_REDUCTION(op) ((op) >= OP_MIN && (op) <= OP_PRODUCT)\n\n')

    out.write('#endif // OPCODES_H\n')


with open("dpu/kernels.h", "w") as out:
    out.write('// GENERATED BY tools/generate_kernels.py. DO NOT EDIT\n\n')
    out.write('#include "./binary.inl"\n')
    out.write('#include "./reduce.inl"\n')
    out.write('#include "./unary.inl"\n')
    out.write('#ifdef PIPELINE\n')
    out.write('#include "./pipeline.inl"\n')
    out.write('#endif\n')
    out.write('#ifndef PIPELINE\n')
    out.write('#define DEFINE_UNIVERSAL_PIPELINE_KERNEL(a)\n')
    out.write('#endif\n\n')

    for id, op in all_ops:
        out.write(f'#define KERNEL_ID_{op.kernel_name().upper()} {id}\n')
    
    # Generate macros (definitions) first
    for id, op in all_ops:
        if op.name == "universal_pipeline":
             out.write('#ifdef PIPELINE\n')
             op.generate_macro(out)
             out.write('#endif\n')
        else:
             op.generate_macro(out)

    out.write(f'#define KERNEL_COUNT {len(all_ops)}\n')
    out.write('int (*kernels[KERNEL_COUNT])(void) = {\n')
    for id, op in all_ops:
        if op.name == "universal_pipeline":
             out.write('#ifdef PIPELINE\n')
             out.write(f'    {op.kernel_name()}, // {id}\n')
             out.write('#else\n')
             out.write(f'    NULL, // {id}\n')
             out.write('#endif\n')
        else:
             out.write(f'    {op.kernel_name()}, // {id}\n')
    out.write('};\n')



with open("host/opinfo.h", "w") as out:
    out.write('#pragma once\n')
    out.write('// GENERATED BY tools/generate_kernels.py. DO NOT EDIT\n\n')
    out.write('template<typename T> struct OpInfo;\n\n')
    for type, group in grouped_ops:
        out.write(f'template<> struct OpInfo<{type}> {{\n')
        # existing binary/reduction
        # existing binary/reduction
        for op in group:
             if not isinstance(op, DPUUnaryOp):
                out.write(f'    static constexpr int {op.name} = {op.kid};\n')
             else:
                out.write(f'    static constexpr int {op.name} = {op.kid};\n')

        
        # Add pipeline opcodes to OpInfo
        for idx, (name, symbol) in enumerate(pipeline_ops):
            if name != 'identity': # optional check
                out.write(f'    static constexpr int {name}_op = {idx};\n')
        
        out.write('};\n\n')

with open("host/kernelids.h", "w") as out:
    out.write('#pragma once\n')
    out.write('// GENERATED BY tools/generate_kernels.py. DO NOT EDIT\n\n')
    out.write('#include <common.h>\n')
    out.write('enum KernelTypes {\n')
    for t in types:
        out.write(f'    KERNEL_TYPE_{t.upper()},\n')
    out.write('};\n')

    out.write(f"enum KernelOperator {{\n")
    for op in ops: # iterating over original ops list to get names
        fake = op('float', 0)
        out.write(f"    KERNEL_OP_{fake.name.upper()},\n")
    for op in ops:
        if isinstance(op('float', 0), DPUBinaryOp):
            fake = op('float', 0)
            out.write(f"    KERNEL_OP_{fake.name.upper()}_SCALAR,\n")
    out.write(f"    KERNEL_OP_PIPELINE,\n")
    out.write("};\n")

    out.write('struct KernelInfo {\n')
    out.write('    int id;\n')
    out.write('    KernelTypes type;\n')
    out.write('    KernelCategory category;\n')
    out.write('    KernelOperator op;\n')
    out.write('    const char *name;\n')
    out.write('};\n')

    # make an array of the info that we have from all_ops
    out.write('static KernelInfo kernel_infos[] = {\n')
    for id, op in all_ops:
        if isinstance(op, DPUBinaryOp):
            category = 'KERNEL_BINARY'
            op_enum = f'KERNEL_OP_{op.name.upper()}'
        elif isinstance(op, DPUUnaryOp):
            category = 'KERNEL_UNARY'
            op_enum = 'KERNEL_OP_PIPELINE'
        elif isinstance(op, DPUReduceOp):
            category = 'KERNEL_REDUCTION'
            op_enum = f'KERNEL_OP_{op.name.upper()}'
        elif isinstance(op, DPUBinaryScalarOp):
            category = 'KERNEL_BINARY_SCALAR'
            op_enum = f'KERNEL_OP_{op.name.upper()}'
        else:
            category = 'KERNEL_UNKNOWN'
            op_enum = 'KERNEL_OP_ADD' # dummy
            
        name = f'"{op.kernel_name().upper()}"'
        out.write(f'    {{ {id}, KERNEL_TYPE_{op.type.upper()}, {category}, {op_enum}, {name} }},\n')
    out.write('};\n')
