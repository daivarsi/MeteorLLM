import pathlib,json,struct,unicodedata
R=pathlib.Path(__file__).resolve().parent;m=json.loads((R/'model/tokenizer.json').read_text(encoding='utf-8'))
tokens=m['tokenizer.ggml.tokens'];ids={t:i for i,t in enumerate(tokens)}
bs=list(range(33,127))+list(range(161,173))+list(range(174,256));cs=bs.copy();n=0
for b in range(256):
 if b not in bs:bs.append(b);cs.append(256+n);n+=1
inverse=dict(zip(map(chr,cs),bs));forward=dict(zip(bs,map(chr,cs)))
blob=bytearray();offsets=[0]
for t in tokens:
 blob.extend(bytes(inverse[c] for c in t) if all(c in inverse for c in t) else t.encode());offsets.append(len(blob))
special=[i for i,t in enumerate(m['tokenizer.ggml.token_type']) if t in (3,4)]
merges=[]
for rank,pair in enumerate(m['tokenizer.ggml.merges']):
 a,b=pair.split(' ');merges.append((ids[a],ids[b],ids[a+b],rank))
merges.sort(key=lambda x:(x[0],x[1]))
h=bytearray(64);h[:8]=b'MTOK0001';out=bytearray(h)
offbase=len(out);out.extend(struct.pack('<'+str(len(offsets))+'I',*offsets))
blobbase=len(out);out.extend(blob)
out.extend(bytes((-len(out))%4));bytebase=len(out);out.extend(struct.pack('<256I',*[ids[forward[b]] for b in range(256)]))
spbase=len(out);out.extend(struct.pack('<'+str(len(special))+'I',*special));out.extend(bytes((-len(out))%8));mergebase=len(out)
for merge in merges:out.extend(struct.pack('<4I',*merge))
struct.pack_into('<9IQ',out,8,len(tokens),len(merges),len(special),offbase,blobbase,len(blob),bytebase,spbase,mergebase,len(out))
(R/'model/tokenizer.bin').write_bytes(out)
lines=['// Generated from Python Unicode '+unicodedata.unidata_version+' general categories.']
for category,name in [('L','letters'),('N','numbers')]:
 ranges=[];start=None
 for cp in range(0x110001):
  hit=cp<0x110000 and unicodedata.category(chr(cp)).startswith(category)
  if hit and start is None:start=cp
  if not hit and start is not None:ranges.append((start,cp-1));start=None
 lines.append('static const uint32_t '+name+'[][2]={'+','.join('{%d,%d}'%r for r in ranges)+'};')
(R/'unicode_ranges.inc').write_text('\n'.join(lines))
print('tokenizer bytes',len(out),'merges',len(merges),'unicode',unicodedata.unidata_version)
