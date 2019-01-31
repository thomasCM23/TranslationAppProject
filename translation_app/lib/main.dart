import 'package:flutter/material.dart';
import 'history.dart';
import 'translation.dart';
import 'globals.dart' as gl;
import 'dart:math';
import 'package:shared_preferences/shared_preferences.dart';
import 'network_call.dart';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Simple Translate',
      theme: ThemeData(
        primarySwatch: Colors.lightGreen,
      ),
      home: MyHomePage(title: 'Simple Translate'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  MyHomePage({Key key, this.title}) : super(key: key);

  // This widget is the home page of your application. It is stateful, meaning
  // that it has a State object (defined below) that contains fields that affect
  // how it looks.

  // This class is the configuration for the state. It holds the values (in this
  // case the title) provided by the parent (in this case the App widget) and
  // used by the build method of the State. Fields in a Widget subclass are
  // always marked "final".

  final String title;

  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  final Translation trans = Translation();
  final TextEditingController _controller = new TextEditingController();
  final _prefKeyId = 'historyID';
  final _prefKeyTxt = 'historyTxt';
  final _prefKeyLang = 'historyLang';

  List<History> historyData = [];
  bool apiCall = false;

  @override
  void initState() {
    trans.translatedText = '';
    apiCall = false;
    _loadHistory();
  }

  _loadHistory() async {
    SharedPreferences his = await SharedPreferences.getInstance();
    List<String> historyDataID = (his.getStringList(_prefKeyId) ?? []);
    List<String> historyDataOriginalTxt =
        (his.getStringList(_prefKeyTxt) ?? []);
    List<String> historyDataLang = (his.getStringList(_prefKeyLang) ?? []);
    setState(() {
      for (int i = 0; i < historyDataID.length; i++) {
        History item = History(
            historyDataOriginalTxt[i], historyDataLang[i], historyDataID[i]);
        historyData.add(item);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    // This method is rerun every time setState is called, for instance as done
    // by the _incrementCounter method above.
    //
    // The Flutter framework has been optimized to make rerunning build methods
    // fast, so that you can just rebuild anything that needs updating rather
    // than having to individually change instances of widgets.
    return Scaffold(
        appBar: AppBar(
          // Here we take the value from the MyHomePage object that was created by
          // the App.build method, and use it to set our appbar title.
          title: Text(widget.title),
        ),
        body: Padding(
          padding: const EdgeInsets.all(12.0),
          child: Center(
            // Center is a layout widget. It takes a single child and positions it
            // in the middle of the parent.
            child: Column(
              children: <Widget>[
                topRow(),
                textFieldContainer(),
                standardPadding(),
                translatedText(),
                standardPadding(),
                historyListBuilder()
              ],
            ),
          ),
        ));
  }

  void onChangeLanguage(String selectedLang) {
    setState(() {
      trans.currentLanguage = selectedLang;
    });
  }

  void onSubmittedText(String txt) async{
    setState(() {
      apiCall = true;
    });
    String text = await fetchTranslation(txt, trans.currentLanguage);
    setState(() {
      apiCall = false;
      trans.translatedText = text;
      String itemId = Random().nextInt(1000).toString() + txt[0];
      historyData.insert(
          0, History(txt, trans.currentLanguage, itemId));
      if (historyData.length > 10) {
        historyData.removeLast();
      }
    });
    updatePrefs();
  }

  void onReloadHistory(String itemId){
    int index = getIndexOfItemInHistory(itemId);
    setState(() {
      trans.currentLanguage = historyData[index].toLang;
      _controller.text = historyData[index].orignalTxt;
    });
    onSubmittedText(historyData[index].orignalTxt);
  }

  void onDeleteHistoryItem(String itemId) {
    int indexToRemove = getIndexOfItemInHistory(itemId);
    setState(() {
      historyData.removeAt(indexToRemove);
    });
    updatePrefs();
  }

  int getIndexOfItemInHistory(String itemId){
    int index;
    for (int i = 0; i < historyData.length; i++) {
      if (historyData[i].itemID == itemId) {
        index = i;
        break;
      }
    }
    return index;
  }

  void updatePrefs() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    List<String> ids = [];
    List<String> txts = [];
    List<String> langs = [];
    for (int i = 0; i < historyData.length; i++) {
      ids.add(historyData[i].itemID);
      txts.add(historyData[i].orignalTxt);
      langs.add(historyData[i].toLang);
    }
    prefs.setStringList(_prefKeyId, ids);
    prefs.setStringList(_prefKeyTxt, txts);
    prefs.setStringList(_prefKeyLang, langs);
  }

  Widget topRow() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: <Widget>[
        Text(
          "English",
          style: TextStyle(fontSize: gl.largeText),
        ),
        Spacer(),
        Icon(Icons.arrow_forward),
        Spacer(),
        DropdownButton(
          value: trans.currentLanguage,
          items: trans.getDropDownMenuItems(),
          onChanged: onChangeLanguage,
        )
      ],
    );
  }

  Widget historyListBuilder() {
    return Expanded(
      child: ListView.separated(
        itemBuilder: (BuildContext context, int index) =>
            HistoryItem(historyData[index], onDeleteHistoryItem, onReloadHistory),
        itemCount: historyData.length,
        separatorBuilder: (context, index) =>
            Divider(color: Colors.white)
      ),
    );
  }

  Widget textFieldContainer() {
    return Stack(
      alignment: const Alignment(1.0, 1.0),
      children: <Widget>[
        TextField(
          decoration: InputDecoration(
            hintText: "Please enter a sentence to translate",
            hintStyle: TextStyle(fontSize: gl.smalltext),
          ),
          autofocus: true,
          onSubmitted: onSubmittedText,
          maxLines: 4,
          controller: _controller,
          keyboardType: TextInputType.text,
          textInputAction: TextInputAction.done,
        ),
        IconButton(
            onPressed: () {
              _controller.clear();
            },
            icon: Icon(Icons.clear))
      ],
    );
  }

  Widget standardPadding() {
    return Padding(
      padding: EdgeInsets.all(10.0),
    );
  }

  Widget translatedText() {
    if(apiCall){
      return CircularProgressIndicator();
    }else{
      return Text(
        trans.translatedText,
        style: new TextStyle(fontSize: gl.normaltext),
      );
    }
  }
}
