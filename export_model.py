"""Losslessly split GGUF tensor bytes into resident layers and streamed embedding.
No weight requantization. Output format is explicitly versioned and bounded.
"""
import pathlib,struct,json,hashlib,sys,math
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
f=src.open('rb')
def u(fmt):return struct.unpack('<'+fmt,f.read(struct.calcsize('<'+fmt)))[0]
def string():return f.read(u('Q')).decode('utf-8')
def value(t):
 if t==8:return string()
 if t==9:
  st,n=u('I'),u('Q');return [value(st) for _ in range(n)]
 return u({0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}[t])
assert f.read(4)==b'GGUF' and u('I')==3
nt,nm=u('Q'),u('Q');meta={}
for _ in range(nm):k=string();meta[k]=value(u('I'))
assert meta['general.architecture']=='qwen3'
tensors=[]
for _ in range(nt):
 name=string();shape=[u('Q') for _ in range(u('I'))];typ=u('I');offset=u('Q')
 assert typ in (0,12,14) and len(shape) in (1,2)
 size=math.prod(shape)*4 if typ==0 else math.prod(shape)//256*{12:144,14:210}[typ]
 tensors.append(dict(name=name,shape=shape,type=typ,source_offset=offset,size=size))
align=meta.get('general.alignment',32);base=(f.tell()+align-1)//align*align
header_size=(128+nt*96+255)//256*256;cursor=header_size
for t in tensors:
 if t['name']=='token_embd.weight':t['offset']=2**64-1
 else:t['offset']=cursor;cursor+=t['size']
h=bytearray(128);h[:8]=b'VQWN0001'
keys=['embedding_length','feed_forward_length','block_count','attention.head_count','attention.head_count_kv','attention.key_length']
vals=[meta['qwen3.'+k] for k in keys]+[len(meta['tokenizer.ggml.tokens']),512]
struct.pack_into('<8I',h,8,*vals)
struct.pack_into('<2f',h,40,meta['qwen3.attention.layer_norm_rms_epsilon'],meta['qwen3.rope.freq_base'])
struct.pack_into('<IIIQ',h,48,nt,header_size,meta['tokenizer.ggml.eos_token_id'],cursor)
records=[]
with (out/'weights.bin').open('wb') as w:
 w.write(h)
 for t in tensors:
  entry=bytearray(96);b=t['name'].encode();assert len(b)<64;entry[:len(b)]=b
  shape=t['shape'];struct.pack_into('<IIIQQ',entry,64,t['type'],shape[0],shape[1] if len(shape)>1 else 1,t['offset'],t['size']);w.write(entry)
 w.write(bytes(header_size-w.tell()))
 for t in tensors:
  f.seek(base+t['source_offset']);data=f.read(t['size']);assert len(data)==t['size']
  if t['offset']==2**64-1:(out/'embedding.bin').write_bytes(data)
  else:assert w.tell()==t['offset'];w.write(data)
  records.append(dict(name=t['name'],size=t['size'],sha256=hashlib.sha256(data).hexdigest()))
 assert w.tell()==cursor
# Decoded byte pieces for output display; full tokenizer metadata retained separately.
bs=list(range(33,127))+list(range(161,173))+list(range(174,256));cs=bs.copy();n=0
for b in range(256):
 if b not in bs:bs.append(b);cs.append(256+n);n+=1
inverse=dict(zip(map(chr,cs),bs))
tokens=meta['tokenizer.ggml.tokens']
with (out/'vocab.bin').open('wb') as v:
 v.write(struct.pack('<I',len(tokens)))
 for token in tokens:
  data=bytes(inverse[c] for c in token) if all(c in inverse for c in token) else token.encode('utf-8')
  v.write(struct.pack('<I',len(data)));v.write(data)
(out/'tokenizer.json').write_text(json.dumps({k:v for k,v in meta.items() if k.startswith('tokenizer.')},ensure_ascii=False),encoding='utf-8')
manifest=dict(source=str(src),source_sha256=hashlib.file_digest(src.open('rb'),'sha256').hexdigest(),resident_bytes=cursor,embedding_bytes=(out/'embedding.bin').stat().st_size,config=vals,tensors=records,conversion='Tensor bytes unchanged; embedding/output storage separated; context capped to 512; FP16 KV planned')
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({k:v for k,v in manifest.items() if k!='tensors'},indent=2))
