import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator, KeyboardAvoidingView, Platform, Pressable,
  ScrollView, StyleSheet, Text, TextInput, View,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';

const API = (process.env.EXPO_PUBLIC_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
const LAB_NAME = { fuel: 'Fuel', stress: 'Stress', movement: 'Rhythm', sleep: 'Sleep' };

async function request(path, options = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  let data = {};
  try { data = await res.json(); } catch (_) {}
  if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
  return data;
}

const fmtShortDate = value => value ? new Date(value).toLocaleDateString([], { month: 'short', day: 'numeric' }) : '';
const fmtDateTime = value => value ? new Date(value).toLocaleString([], { weekday:'short', month:'short', day:'numeric', hour:'numeric', minute:'2-digit' }) : '';
const titleDate = value => value ? new Date(value).toLocaleDateString([], { weekday:'long', month:'long', day:'numeric' }) : 'Today';

function Loading() { return <ActivityIndicator style={styles.loader} />; }
function ErrorText({ children }) { return children ? <Text style={styles.error}>{children}</Text> : null; }
function SectionTitle({ children }) { return <Text style={styles.sectionTitle}>{children}</Text>; }
function Pill({ children, tone='plain' }) { return <View style={[styles.pill, styles[`pill_${tone}`]]}><Text style={styles.pillText}>{children}</Text></View>; }
function Card({ children, style }) { return <View style={[styles.card, style]}>{children}</View>; }
function Divider() { return <View style={styles.divider}/>; }

function Header({ pid }) {
  return <View style={styles.header}>
    <View><Text style={styles.brand}>BODY LAB</Text><Text style={styles.tagline}>your personal scientist</Text></View>
    <Text style={styles.pid}>{pid}</Text>
  </View>;
}

function LabGrid({ labs }) {
  if (!labs?.length) return null;
  return <View style={styles.labGrid}>{labs.map(l => <View key={l.key} style={styles.labCard}>
    <Text style={styles.labName}>{l.name}</Text>
    <Text style={styles.labSituation}>{l.situation}s</Text>
    {l.key === 'stress' ? <Text style={styles.labNote}>stress 1–10, personal</Text> : null}
    <View style={styles.labFooter}><Text style={styles.labCount}>{l.cards} cards</Text><Text style={styles.labOpen}>{l.open} open</Text></View>
  </View>)}</View>;
}

function FeedCard({ item }) {
  const [open, setOpen] = useState(false);
  const kind = item.kind === 'discovery' ? 'Discovery confirmed' : item.kind === 'case' ? 'Case solved' : item.kind === 'rejected' ? 'Not confirmed' : 'Fading';
  const tone = item.kind === 'discovery' ? 'ok' : item.kind === 'case' ? 'case' : 'plain';
  return <Pressable onPress={() => setOpen(v => !v)} style={({pressed}) => [styles.feedCard, pressed && styles.pressed]}>
    <View style={styles.feedMeta}><View style={styles.feedMetaLeft}><Pill tone={tone}>{kind}</Pill><Text style={styles.feedLab}>{LAB_NAME[item.lab] || item.lab}</Text></View><Text style={styles.feedDate}>{fmtShortDate(item.ts)}</Text></View>
    <Text style={styles.feedTitle} numberOfLines={open ? undefined : 2}>{item.title}</Text>
    {open ? <><Text style={styles.feedBody}>{item.body}</Text><Text style={styles.collapse}>Tap to collapse</Text></> : <Text style={styles.expand}>View details  ›</Text>}
  </Pressable>;
}

function Today({ pid }) {
  const [data,setData]=useState(null); const [error,setError]=useState('');
  useEffect(()=>{ request(`/api/today/${pid}`).then(setData).catch(e=>setError(e.message)); },[pid]);
  if (!data && !error) return <Loading/>;
  const rank=data?.rank || {};
  return <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
    <Text style={styles.pageTitle}>{titleDate(data?.as_of)}</Text><ErrorText>{error}</ErrorText>

    <SectionTitle>Your labs</SectionTitle>
    <LabGrid labs={data?.labs}/>

    <SectionTitle>Recent</SectionTitle>
    {data?.messages?.length ? data.messages.map((m,i)=><FeedCard key={`${m.ts}-${i}`} item={m}/>) : <Card><Text style={styles.eyebrow}>NOTHING TO REPORT YET</Text><Text style={styles.body}>The agent only writes when it has an answer: a solved case or a confirmed discovery.</Text></Card>}

    {data && <Card><Text style={styles.eyebrow}>CLOSED QUIETLY THIS WEEK</Text><Text style={styles.body}>{data.closed_quietly.unexplained} surprise{data.closed_quietly.unexplained===1?'':'s'} with no clear reason · {data.closed_quietly.bad_data} dismissed as bad data</Text><Text style={styles.muted}>No alerts were sent for these. Details are in the Notebook.</Text></Card>}

    {data && <><SectionTitle>Scientist rank</SectionTitle><Card><View style={styles.rankRow}><Text style={styles.rankName}>{rank.name}</Text><Text style={styles.rankPoints}>{rank.points} points</Text></View><Text style={styles.muted}>{rank.next_rank_points ? `${rank.next_rank_points-rank.points} points to the next rank` : 'Top rank reached'}</Text></Card></>}

    {data?.quests?.length ? <><SectionTitle>Quests</SectionTitle>{data.quests.map((q,i)=><Card key={i}><Text style={styles.eyebrow}>OPTIONAL · DETECTED AUTOMATICALLY</Text><Text style={styles.compactTitle}>{q.title}</Text><Text style={styles.body}>{q.done?'Done':`${q.progress} of ${q.target}`}</Text></Card>)}</> : null}
  </ScrollView>;
}

function CaseListItem({ item, selected, onPress }) {
  const verdict = item.verdict==='lead'?'Solved':item.verdict==='unexplained'?'Unexplained':item.verdict==='bad_data'?'Bad data':item.verdict;
  return <Pressable onPress={onPress} style={({pressed})=>[styles.listCard, selected&&styles.listCardSelected, pressed&&styles.pressed]}>
    <View style={styles.listTop}><Text style={styles.listMeta}>{(LAB_NAME[item.lab]||item.lab).toUpperCase()} · {String(verdict).toUpperCase()}</Text><Text style={styles.listDate}>{fmtShortDate(item.ts)}</Text></View>
    <Text style={styles.listTitle} numberOfLines={2}>{item.title || item.message || 'Case'}</Text>
    <Text style={styles.expand}>View case  ›</Text>
  </Pressable>;
}

function Cases({ pid }) {
  const [items,setItems]=useState(null); const [selected,setSelected]=useState(null); const [detail,setDetail]=useState(null); const [error,setError]=useState('');
  useEffect(()=>{ request(`/api/cases/${pid}`).then(d=>setItems(d.cases)).catch(e=>setError(e.message)); },[pid]);
  useEffect(()=>{ if(selected) request(`/api/case/${pid}/${encodeURIComponent(selected)}`).then(d=>setDetail(d.case)).catch(e=>setError(e.message)); },[pid,selected]);
  if (!items && !error) return <Loading/>;
  const verdict = v => v==='lead'?'Solved':v==='unexplained'?'Unexplained':v==='bad_data'?'Bad data':v;

  if (selected) return <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
    <Pressable onPress={()=>{setSelected(null);setDetail(null);}}><Text style={styles.back}>‹ Cases</Text></Pressable>
    {!detail ? <Loading/> : <>
      <View style={styles.detailMeta}><Pill tone="accent">{detail.lab_name} Lab</Pill><Pill tone={detail.verdict==='lead'?'case':'plain'}>{verdict(detail.verdict)}</Pill></View>
      <Text style={styles.detailTitle}>{detail.title || 'Case'}</Text>
      <Text style={styles.detailLead}>{detail.message}</Text>
      <Text style={styles.detailId}>{detail.event_id} · agent: {detail.agent || ''}</Text>
      {detail.tools?.length ? <DetailSection label="AGENT STEPS"><Text style={styles.body}>{detail.tools.join(' → ')}</Text></DetailSection> : null}
      {detail.checks?.length ? <DetailSection label="STEP 1 · DATA CHECK">{detail.checks.map((c,i)=><View key={i} style={styles.detailLine}><Text style={[styles.checkMark,c.passed?styles.pass:styles.fail]}>{c.passed?'✓':'✕'}</Text><View style={styles.detailLineText}><Text style={styles.compactTitle}>{c.check}</Text><Text style={styles.muted}>{c.detail}</Text></View></View>)}</DetailSection> : null}
      {detail.differences?.length ? <DetailSection label="STEP 2 · WHAT WAS DIFFERENT">{detail.differences.slice(0,5).map((d,i)=><View key={i} style={styles.difference}><View style={styles.listTop}><Text style={styles.compactTitle}>{d.label}</Text><Pill tone={d.level==='very unusual'?'case':d.level==='somewhat'?'warn':'plain'}>{d.level}</Pill></View><Text style={styles.muted}>this time {d.this_time_text ?? d.this_time} · typical {d.typical_text ?? d.similar_median}</Text></View>)}</DetailSection> : null}
      {detail.hypothesis ? <DetailSection label="STEP 3 · VERDICT"><Text style={styles.compactTitle}>{detail.hypothesis.hyp_id}: {detail.hypothesis.claim}</Text><Text style={styles.body}>Status: {detail.hypothesis.status}</Text><Text style={styles.muted}>{detail.hypothesis.supports} supporting · {detail.hypothesis.contradicts} against</Text></DetailSection> : null}
    </>}
  </ScrollView>;

  return <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}><Text style={styles.pageTitle}>Cases</Text><ErrorText>{error}</ErrorText>
    {!items?.length ? <Text style={styles.body}>No surprises investigated yet.</Text> : items.map(e=><CaseListItem key={e.event_id} item={e} onPress={()=>setSelected(e.event_id)}/>)}
  </ScrollView>;
}

function DetailSection({ label, children }) { return <View style={styles.detailSection}><Text style={styles.eyebrow}>{label}</Text><Divider/>{children}</View>; }

function DiscoveryCard({ d }) {
  const [open,setOpen]=useState(false);
  return <Pressable onPress={()=>setOpen(v=>!v)} style={({pressed})=>[styles.listCard,d.rarity==='legendary'&&styles.goldEdge,d.rarity==='rare'&&styles.blueEdge,pressed&&styles.pressed]}>
    <View style={styles.listTop}><Text style={styles.listMeta}>{String(d.rarity).toUpperCase()} · {String(d.lab_name).toUpperCase()}</Text>{d.status==='fading'?<Pill tone="warn">Fading</Pill>:null}</View>
    <Text style={styles.listTitle}>{d.title}</Text>
    {d.effect_text ? <Text style={styles.effect}>{d.effect_text}</Text> : null}
    {open ? <><Text style={styles.body}>{d.claim}</Text><Text style={styles.muted}>{d.evidence}</Text><Text style={styles.collapse}>Tap to collapse</Text></> : <Text style={styles.expand}>View discovery  ›</Text>}
  </Pressable>;
}

function Discoveries({ pid }) {
  const [data,setData]=useState(null); const [error,setError]=useState('');
  useEffect(()=>{request(`/api/discoveries/${pid}`).then(setData).catch(e=>setError(e.message));},[pid]);
  if(!data&&!error)return <Loading/>;
  return <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}><Text style={styles.pageTitle}>Discoveries</Text><Text style={styles.pageSub}>{data?.discoveries?.length||0} collected</Text><ErrorText>{error}</ErrorText>
    {data && <View style={styles.countRow}><View style={styles.countBox}><Text style={styles.countNum}>{data.counts.legendary}</Text><Text style={styles.countLabel}>Legendary</Text></View><View style={styles.countBox}><Text style={styles.countNum}>{data.counts.rare}</Text><Text style={styles.countLabel}>Rare</Text></View><View style={styles.countBox}><Text style={styles.countNum}>{data.counts.common}</Text><Text style={styles.countLabel}>Common</Text></View></View>}
    {data?.discoveries?.map((d,i)=><DiscoveryCard key={d.card_id||i} d={d}/>) }
    {data?.close?.length ? <><SectionTitle>Close to a discovery</SectionTitle>{data.close.map((h,i)=><Card key={`close-${i}`}><Text style={styles.compactTitle}>{LAB_NAME[h.lab]||h.lab} Lab · {h.hyp_id}</Text><Text style={styles.muted}>{Math.max(3-(h.supports||0),1)} test away</Text></Card>)}</> : null}
    {data?.rejected_titles?.length ? <Text style={styles.muted}>Withdrawn after newer data disagreed: {data.rejected_titles.join(', ')}</Text> : null}
  </ScrollView>;
}

function HypothesisCard({ h }) {
  const [open,setOpen]=useState(false);
  return <Pressable onPress={()=>setOpen(v=>!v)} style={({pressed})=>[styles.listCard,pressed&&styles.pressed]}>
    <View style={styles.listTop}><Text style={styles.listMeta}>{h.hyp_id} · {String(h.lab_name).toUpperCase()}</Text><Pill tone={h.status==='confirmed'?'ok':h.status==='rejected'?'case':h.status==='fading'?'warn':'plain'}>{h.status}</Pill></View>
    <Text style={styles.listTitle}>{h.claim}</Text>
    <Text style={styles.evidenceLine}>{h.supports} supporting · {h.contradicts} against</Text>
    {open ? <><Text style={styles.muted}>{h.chances} chances</Text><Text style={styles.collapse}>Tap to collapse</Text></> : <Text style={styles.expand}>View evidence  ›</Text>}
  </Pressable>;
}

function Notebook({ pid }) {
  const [data,setData]=useState(null); const [error,setError]=useState('');
  useEffect(()=>{request(`/api/notebook/${pid}`).then(setData).catch(e=>setError(e.message));},[pid]);
  if(!data&&!error)return <Loading/>;
  const f=data?.funnel||{};
  return <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}><Text style={styles.pageTitle}>Notebook</Text><Text style={styles.pageSub}>The agent's lab notebook</Text><ErrorText>{error}</ErrorText>
    {data && <Text style={styles.eyebrow}>HYPOTHESES · {data.open_slots} OF 10 OPEN SLOTS · MEAL-RELATED {data.meal_related_open} OF 2 MAX</Text>}
    {data?.hypotheses?.map((h,i)=><HypothesisCard key={h.hyp_id||i} h={h}/>) }
    {data && <><SectionTitle>Every surprise so far</SectionTitle><Card>{[['Surprising events','surprises'],['Bad data, dismissed','bad_data'],['No clear reason','unexplained'],['Became leads','leads'],['Confirmed discoveries','discoveries']].map(([label,key],i)=><React.Fragment key={key}><View style={styles.statRow}><Text style={styles.body}>{label}</Text><Text style={styles.stat}>{f[key]||0}</Text></View>{i<4?<Divider/>:null}</React.Fragment>)}<Text style={styles.muted}>Most surprises are noise or bad data. Only repeated patterns become discoveries.</Text></Card><Card><Text style={styles.eyebrow}>SITUATIONS WATCHED</Text><Text style={styles.rankName}>{data.situations_watched}</Text><Text style={styles.body}>{data.situations_good_data} passed the data check and counted as natural experiments.</Text></Card></>}
  </ScrollView>;
}

function Chat({ pid }) {
  const starters=['What have you learned about me so far?','Which hypothesis has the strongest evidence?','Have any of your ideas been proven wrong?','What should Body Lab investigate next?'];
  const [messages,setMessages]=useState([]); const [text,setText]=useState(''); const [sending,setSending]=useState(false); const scroll=useRef(null);
  const send=async raw=>{const message=(raw??text).trim();if(!message||sending)return;const history=messages.slice(-8);const next=[...messages,{role:'user',content:message}];setMessages(next);setText('');setSending(true);try{const d=await request('/api/chat',{method:'POST',body:JSON.stringify({participant_id:pid,message,history})});setMessages([...next,{role:'assistant',content:d.answer}]);}catch(e){setMessages([...next,{role:'assistant',content:`I couldn't query the Body Lab notebook right now. ${e.message}`}]);}finally{setSending(false);setTimeout(()=>scroll.current?.scrollToEnd({animated:true}),50);}};
  return <KeyboardAvoidingView style={styles.flex} behavior={Platform.OS==='ios'?'padding':undefined} keyboardVerticalOffset={86}><ScrollView ref={scroll} contentContainerStyle={styles.chatContent} onContentSizeChange={()=>scroll.current?.scrollToEnd({animated:true})} showsVerticalScrollIndicator={false}><Text style={styles.pageTitle}>Ask Body Lab</Text><Text style={styles.pageSub}>Ask about patterns, discoveries, hypotheses, and cases Body Lab has actually observed in your data.</Text>{!messages.length?<><Text style={styles.eyebrow}>TRY ASKING</Text>{starters.map(q=><Pressable key={q} style={styles.starter} onPress={()=>send(q)}><Text style={styles.starterText}>{q}</Text><Text style={styles.chevron}>›</Text></Pressable>)}</>:null}{messages.map((m,i)=><View key={i} style={[styles.bubble,m.role==='user'?styles.userBubble:styles.botBubble]}><Text style={[styles.bubbleText,m.role==='user'&&styles.userBubbleText]}>{m.content}</Text></View>)}{sending?<View style={[styles.bubble,styles.botBubble]}><ActivityIndicator/></View>:null}</ScrollView><View style={styles.composer}><TextInput value={text} onChangeText={setText} placeholder="Ask about your Body Lab data…" style={styles.input} multiline/><Pressable onPress={()=>send()} style={styles.send}><Text style={styles.sendText}>↑</Text></Pressable></View></KeyboardAvoidingView>;
}

const TABS=[{key:'today',label:'Today',icon:'●'},{key:'case',label:'Case',icon:'◇'},{key:'discoveries',label:'Discoveries',icon:'★'},{key:'notebook',label:'Notebook',icon:'▤'},{key:'chat',label:'Ask',icon:'◉'}];

function BottomNav({ tab, setTab }) {
  return <View style={styles.tabbar}>{TABS.map(t=><Pressable key={t.key} onPress={()=>setTab(t.key)} style={styles.tab}><Text style={[styles.tabIcon,tab===t.key&&styles.tabActive]}>{t.icon}</Text><Text numberOfLines={1} style={[styles.tabLabel,tab===t.key&&styles.tabActive]}>{t.label}</Text></Pressable>)}</View>;
}

export default function App(){
  const[tab,setTab]=useState('today'); const[pid,setPid]=useState('S01'); const[bootError,setBootError]=useState('');
  useEffect(()=>{request('/api/participants').then(d=>{if(d.participants?.length)setPid(d.participants[0]);}).catch(e=>setBootError(e.message));},[]);
  const screen=useMemo(()=>({today:<Today pid={pid}/>,case:<Cases pid={pid}/>,discoveries:<Discoveries pid={pid}/>,notebook:<Notebook pid={pid}/>,chat:<Chat pid={pid}/>})[tab],[tab,pid]);
  return <View style={styles.safe}><StatusBar style="dark"/><Header pid={pid}/><View style={styles.screen}>{bootError?<View style={styles.content}><ErrorText>Can't reach the Body Lab API at {API}. {bootError}</ErrorText></View>:screen}</View><BottomNav tab={tab} setTab={setTab}/></View>;
}

const C={ink:'#14292E',text:'#506267',muted:'#879693',line:'#DDE5E2',bg:'#F5F8F7',white:'#FFFFFF',blue:'#315EC9',blueSoft:'#E8EEFC',green:'#2E8558',greenSoft:'#E3F2E9',orange:'#C85B37',orangeSoft:'#FBE9E2',gold:'#9A700B',warn:'#9A6B20'};
const styles=StyleSheet.create({
  safe:{flex:1,backgroundColor:C.bg,paddingTop:Platform.OS==='ios'?50:18},screen:{flex:1},flex:{flex:1},
  header:{height:76,paddingHorizontal:20,flexDirection:'row',alignItems:'center',justifyContent:'space-between',borderBottomWidth:1,borderBottomColor:C.line,backgroundColor:C.white},brand:{fontSize:21,fontWeight:'900',letterSpacing:3.2,color:C.ink},tagline:{fontSize:12,color:C.muted,marginTop:3},pid:{fontSize:14,fontWeight:'800',color:C.text},
  content:{paddingHorizontal:18,paddingTop:20,paddingBottom:28,gap:10},chatContent:{paddingHorizontal:18,paddingTop:20,paddingBottom:18,gap:10},
  pageTitle:{fontSize:30,lineHeight:35,fontWeight:'850',color:C.ink,letterSpacing:-0.7},pageSub:{fontSize:14,lineHeight:20,color:C.muted,marginTop:-5,marginBottom:4},sectionTitle:{fontSize:15,fontWeight:'800',color:C.ink,marginTop:9,marginBottom:1},
  card:{backgroundColor:C.white,borderWidth:1,borderColor:C.line,borderRadius:14,padding:14,gap:7},pressed:{opacity:.72},divider:{height:1,backgroundColor:'#EDF1EF',marginVertical:5},
  eyebrow:{fontSize:9.5,fontWeight:'850',letterSpacing:1.15,color:C.muted},body:{fontSize:13.5,lineHeight:19.5,color:C.text},muted:{fontSize:11.5,lineHeight:16.5,color:C.muted},compactTitle:{fontSize:14,fontWeight:'750',lineHeight:19,color:C.ink},error:{color:'#B24747',fontSize:14,lineHeight:20},loader:{marginTop:45},
  pill:{paddingHorizontal:7,paddingVertical:3,borderRadius:99,backgroundColor:'#F0F4F2'},pill_ok:{backgroundColor:C.greenSoft},pill_case:{backgroundColor:C.orangeSoft},pill_accent:{backgroundColor:C.blueSoft},pill_warn:{backgroundColor:'#F8EDD8'},pill_gold:{backgroundColor:'#F7EFD5'},pill_plain:{backgroundColor:'#F0F4F2'},pillText:{fontSize:9,fontWeight:'800',color:C.text},
  labGrid:{flexDirection:'row',flexWrap:'wrap',gap:9},labCard:{width:'48.6%',minHeight:118,backgroundColor:C.white,borderWidth:1,borderColor:C.line,borderRadius:14,padding:13},labName:{fontSize:17,fontWeight:'850',color:C.ink},labSituation:{fontSize:12.5,lineHeight:17,color:C.text,marginTop:5},labNote:{fontSize:10.5,color:C.muted,marginTop:2},labFooter:{marginTop:'auto',paddingTop:12,flexDirection:'row',justifyContent:'space-between'},labCount:{fontSize:10.5,fontWeight:'750',color:C.blue},labOpen:{fontSize:10.5,color:C.muted},
  feedCard:{backgroundColor:C.white,borderWidth:1,borderColor:C.line,borderRadius:14,padding:14,gap:7},feedMeta:{flexDirection:'row',justifyContent:'space-between',alignItems:'center',gap:8},feedMetaLeft:{flexDirection:'row',alignItems:'center',gap:7,flexShrink:1},feedLab:{fontSize:9.5,fontWeight:'850',color:C.muted,textTransform:'uppercase'},feedDate:{fontSize:10.5,color:C.muted},feedTitle:{fontSize:16,lineHeight:21,fontWeight:'800',color:C.ink},feedBody:{fontSize:13,lineHeight:19,color:C.text},expand:{fontSize:11.5,fontWeight:'750',color:C.blue,marginTop:1},collapse:{fontSize:10.5,color:C.muted,marginTop:1},
  rankRow:{flexDirection:'row',alignItems:'baseline',justifyContent:'space-between'},rankName:{fontSize:23,fontWeight:'850',color:C.ink},rankPoints:{fontSize:12,fontWeight:'750',color:C.blue},
  listCard:{backgroundColor:C.white,borderWidth:1,borderColor:C.line,borderRadius:14,padding:14,gap:7},listCardSelected:{borderColor:C.blue},listTop:{flexDirection:'row',justifyContent:'space-between',alignItems:'center',gap:8},listMeta:{fontSize:9.5,fontWeight:'850',letterSpacing:.75,color:C.muted,flexShrink:1},listDate:{fontSize:10.5,color:C.muted},listTitle:{fontSize:16,lineHeight:21,fontWeight:'800',color:C.ink},blueEdge:{borderLeftWidth:3,borderLeftColor:C.blue},goldEdge:{borderLeftWidth:3,borderLeftColor:C.gold},effect:{fontSize:20,fontWeight:'850',color:C.ink},evidenceLine:{fontSize:12,color:C.text},
  back:{fontSize:14,fontWeight:'750',color:C.blue,marginBottom:5},detailMeta:{flexDirection:'row',gap:6,flexWrap:'wrap'},detailTitle:{fontSize:25,lineHeight:30,fontWeight:'850',color:C.ink,letterSpacing:-.4},detailLead:{fontSize:15,lineHeight:22,color:C.text},detailId:{fontSize:10.5,color:C.muted},detailSection:{backgroundColor:C.white,borderWidth:1,borderColor:C.line,borderRadius:14,padding:14,gap:8,marginTop:2},detailLine:{flexDirection:'row',gap:9,alignItems:'flex-start'},detailLineText:{flex:1},checkMark:{fontSize:14,fontWeight:'900'},pass:{color:C.green},fail:{color:'#B24747'},difference:{gap:5,paddingVertical:3},
  countRow:{flexDirection:'row',gap:8},countBox:{flex:1,backgroundColor:C.white,borderWidth:1,borderColor:C.line,borderRadius:12,paddingVertical:11,alignItems:'center'},countNum:{fontSize:20,fontWeight:'850',color:C.ink},countLabel:{fontSize:9.5,fontWeight:'750',color:C.muted,marginTop:2},statRow:{flexDirection:'row',justifyContent:'space-between',alignItems:'center',paddingVertical:2},stat:{fontSize:15,fontWeight:'850',color:C.ink},
  starter:{borderWidth:1,borderColor:C.line,borderRadius:13,paddingHorizontal:13,paddingVertical:12,backgroundColor:C.white,flexDirection:'row',alignItems:'center',gap:8},starterText:{fontSize:13.5,lineHeight:18,color:C.ink,fontWeight:'650',flex:1},chevron:{fontSize:20,color:C.muted},bubble:{maxWidth:'88%',borderRadius:15,paddingHorizontal:13,paddingVertical:10},botBubble:{alignSelf:'flex-start',backgroundColor:C.white,borderWidth:1,borderColor:C.line},userBubble:{alignSelf:'flex-end',backgroundColor:C.ink},bubbleText:{fontSize:13.5,lineHeight:19,color:C.text},userBubbleText:{color:C.white},composer:{paddingHorizontal:12,paddingVertical:8,borderTopWidth:1,borderTopColor:C.line,backgroundColor:C.white,flexDirection:'row',alignItems:'flex-end',gap:7},input:{flex:1,minHeight:40,maxHeight:96,borderWidth:1,borderColor:'#D7DFDC',borderRadius:20,paddingHorizontal:13,paddingTop:10,paddingBottom:8,fontSize:13.5,color:C.ink},send:{width:40,height:40,borderRadius:20,backgroundColor:C.ink,alignItems:'center',justifyContent:'center'},sendText:{color:C.white,fontSize:20,fontWeight:'700'},
  tabbar:{height:68,paddingBottom:Platform.OS==='ios'?8:4,borderTopWidth:1,borderTopColor:C.line,backgroundColor:C.white,flexDirection:'row',alignItems:'center'},tab:{flex:1,height:58,alignItems:'center',justifyContent:'center',gap:2,paddingHorizontal:2},tabIcon:{fontSize:15,color:'#9AA7A4'},tabLabel:{fontSize:8.5,fontWeight:'700',color:'#879693',maxWidth:72},tabActive:{color:C.blue},
});
