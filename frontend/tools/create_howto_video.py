from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import subprocess

OUT = Path(__file__).resolve().parents[1] / 'public' / 'how-to-use'
OUT.mkdir(parents=True, exist_ok=True)
W, H = 1280, 720
BG = '#0b0f14'
SURFACE = '#111720'
SURFACE2 = '#151c24'
BORDER = '#27313c'
TEXT = '#e5e7eb'
MUTED = '#8993a0'
BLUE = '#5cc8ff'
AMBER = '#d4a15c'
GREEN = '#70c7a4'
RED = '#d27878'
VIOLET = '#9e91c7'

FONT_DIR = '/usr/share/fonts/truetype/dejavu'
FONT = f'{FONT_DIR}/DejaVuSans.ttf'
BOLD = f'{FONT_DIR}/DejaVuSans-Bold.ttf'
MONO = f'{FONT_DIR}/DejaVuSansMono.ttf'

def font(path, size):
    return ImageFont.truetype(path, size)

def text(draw, xy, value, size=24, fill=TEXT, bold=False, mono=False):
    path = MONO if mono else (BOLD if bold else FONT)
    draw.text(xy, value, font=font(path, size), fill=fill)

def panel(draw, box, fill=SURFACE, outline=BORDER, radius=4):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2)

def header(draw, step, title, subtitle):
    text(draw, (72, 52), 'CONTRASTIVE SPEECH ANALYTICS', 15, BLUE, bold=True, mono=True)
    text(draw, (72, 101), title, 42, TEXT, bold=True)
    text(draw, (72, 164), subtitle, 20, MUTED)
    text(draw, (1040, 60), f'{step:02d} / 06', 15, MUTED, mono=True)
    draw.line((72, 205, 1208, 205), fill=BORDER, width=2)

def footer(draw, label):
    draw.line((72, 665, 1208, 665), fill=BORDER, width=2)
    text(draw, (72, 683), label, 13, MUTED, mono=True)
    text(draw, (1050, 683), 'DEMO MODE / MOCK DATA', 13, AMBER, mono=True)

def slide_1():
    im = Image.new('RGB', (W,H), BG); d=ImageDraw.Draw(im)
    text(d, (72, 60), 'HOW TO USE', 15, BLUE, bold=True, mono=True)
    text(d, (72, 145), 'See exactly where\nyour delivery changes.', 58, TEXT, bold=True)
    text(d, (76, 310), 'A short walkthrough of the speech-analysis workspace.', 22, MUTED)
    panel(d, (72, 410, 1208, 590), SURFACE)
    steps = [('01','UPLOAD',BLUE),('02','ALIGN',GREEN),('03','ANALYZE',AMBER),('04','IMPROVE',VIOLET)]
    for i,(num,label,color) in enumerate(steps):
        x=120+i*270
        text(d,(x,445),num,16,color,mono=True)
        text(d,(x,480),label,24,TEXT,bold=True)
        if i<3: d.line((x+120,467,x+225,467),fill=BORDER,width=2)
    footer(d,'01  ORIENTATION')
    return im

def slide_2():
    im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im);header(d,2,'Start with Evaluate','Upload the delivery you want to inspect, then choose a matching reference.')
    panel(d,(72,245,590,595)); panel(d,(620,245,1208,595))
    for x,label,sub,color in [(105,'PARTICIPANT AUDIO','Your recording',AMBER),(653,'REFERENCE AUDIO','Ideal or baseline delivery',BLUE)]:
        text(d,(x,277),label,13,color,bold=True,mono=True); text(d,(x,318),sub,25,TEXT,bold=True)
        d.rectangle((x,380,x+480,505),outline=BORDER,width=2)
        text(d,(x+28,414),'Drop audio here',22,MUTED)
        text(d,(x+28,458),'or browse a file',15,BLUE)
        text(d,(x,535),'Filename, duration, and format appear after validation.',14,MUTED)
    footer(d,'02  INPUTS')
    return im

def slide_3():
    im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im);header(d,3,'Confirm the same text','The analysis is contrastive: both recordings must follow the same transcript.')
    panel(d,(72,245,1208,575))
    text(d,(105,278),'SPEECH TRANSCRIPT',13,BLUE,bold=True,mono=True)
    text(d,(105,318),'Every generation changes the way we communicate.',26,TEXT,bold=True)
    text(d,(105,365),'The next generation of communication will be defined by clarity and care.',22,TEXT)
    d.line((105,430,1175,430),fill=BORDER,width=2)
    text(d,(105,464),'MATCH STATUS',13,MUTED,mono=True); text(d,(250,461),'Exact transcript pairing',18,GREEN,bold=True)
    text(d,(105,510),'ALIGNMENT',13,MUTED,mono=True); text(d,(250,507),'Ready for comparison',18,TEXT)
    footer(d,'03  ALIGNMENT')
    return im

def slide_4():
    im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im);header(d,4,'Run the analysis','The pipeline moves from audio validation to grounded feedback.')
    panel(d,(72,245,1208,575))
    stages=['PREPARING AUDIO','ALIGNING SPEECH','EXTRACTING FEATURES','COMPARING DELIVERY','BUILDING FEEDBACK']
    for i,label in enumerate(stages):
        y=288+i*52
        color=GREEN if i<4 else AMBER
        d.ellipse((108,y+4,124,y+20),fill=color)
        text(d,(150,y),label,18,TEXT,bold=True,mono=True)
        text(d,(720,y+1), 'complete' if i<4 else 'current', 14, color, mono=True)
        if i<len(stages)-1: d.line((116,y+21,116,y+52),fill=BORDER,width=2)
    text(d,(150,540),'No fake percentage progress. The current stage stays visible while the API works.',15,MUTED)
    footer(d,'04  PROCESS')
    return im

def slide_5():
    im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im);header(d,5,'Select a flaw on the timeline','Every finding is linked to a timestamp, feature, transcript region, and explanation.')
    panel(d,(72,250,1208,565))
    text(d,(105,276),'BASELINE / IDEAL',12,BLUE,mono=True); text(d,(105,300),'PARTICIPANT',12,VIOLET,mono=True)
    # waveform
    for row, color, shift in [(0,BLUE,0),(1,VIOLET,18)]:
        base=360+shift
        for i in range(96):
            x=105+i*10.9; amp=15+abs(__import__('math').sin(i*.38))*32
            d.line((x,base-amp,x,base+amp),fill=color,width=2)
    flaws=[(330,30,'01',AMBER),(520,43,'02',RED),(840,32,'03',AMBER)]
    for x,w,num,color in flaws:
        d.rectangle((x,270,x+w,470),outline=color,width=2)
        text(d,(x+5,277),num,12,color,mono=True)
    d.line((730,260,730,495),fill=TEXT,width=2)
    text(d,(105,510),'00:00',13,MUTED,mono=True); text(d,(400,510),'00:10',13,MUTED,mono=True); text(d,(700,510),'00:20',13,MUTED,mono=True); text(d,(1000,510),'00:30',13,MUTED,mono=True)
    text(d,(105,540),'Click a marked region to move playback and focus the evidence panel.',15,TEXT)
    footer(d,'05  GROUND')
    return im

def slide_6():
    im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im);header(d,6,'Read the evidence, then improve','Scores summarize the evidence. The timestamped finding tells you what to practice.')
    panel(d,(72,245,560,575)); panel(d,(590,245,1208,575))
    text(d,(105,278),'SELECTED FINDING',13,AMBER,bold=True,mono=True)
    text(d,(105,322),'PACING DEVIATION',25,TEXT,bold=True)
    text(d,(105,365),'00:18.42 - 00:21.10',16,AMBER,mono=True)
    text(d,(105,420),'Speech rate',13,MUTED,mono=True); text(d,(380,420),'3.8 → 5.1 syllables/sec',15,TEXT,mono=True)
    text(d,(105,460),'Deviation',13,MUTED,mono=True); text(d,(380,460),'+34.2%',16,RED,bold=True,mono=True)
    text(d,(105,500),'Confidence',13,MUTED,mono=True); text(d,(380,500),'91%',16,GREEN,bold=True,mono=True)
    text(d,(625,278),'OVERALL EVALUATION',13,BLUE,bold=True,mono=True)
    text(d,(625,320),'82',56,TEXT,bold=True); text(d,(705,347),'/ 100',21,MUTED)
    d.line((625,405,1168,405),fill=BORDER,width=2)
    text(d,(625,435),'Pacing  68',16,TEXT); text(d,(625,470),'Pause Control  82',16,TEXT); text(d,(625,505),'Energy  84',16,TEXT)
    text(d,(625,540),'Use the limitation notes to interpret coverage honestly.',14,MUTED)
    footer(d,'06  EXPLAIN / SCORE')
    return im

slides=[slide_1(),slide_2(),slide_3(),slide_4(),slide_5(),slide_6()]
for i, im in enumerate(slides,1):
    im.save(OUT/f'slide-{i:02d}.png')

concat = OUT/'concat.txt'
with concat.open('w') as f:
    for i in range(1,7):
        f.write(f"file 'slide-{i:02d}.png'\n")
        f.write('duration 4\n')
    f.write("file 'slide-06.png'\n")

mp4 = Path(__file__).resolve().parents[1] / 'public' / 'how-to-use.mp4'
subprocess.run(['ffmpeg','-y','-f','concat','-safe','0','-i',str(concat),'-vf','format=yuv420p','-r','30','-movflags','+faststart',str(mp4)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(mp4)
