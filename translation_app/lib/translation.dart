import 'package:flutter/material.dart';
import 'globals.dart' as gl;

class Translation {
  final List<String> _languages = ["French", "Italin", "German"];
  String currentLanguage;
  String translatedText;

  Translation() {
    this.currentLanguage = getFirstLanguage();
  }

  List<DropdownMenuItem<String>> getDropDownMenuItems() {
    List<DropdownMenuItem<String>> items = new List();
    for (String lang in _languages) {
      items.add(DropdownMenuItem(
        value: lang,
        child: Text(lang, style: new TextStyle(fontSize: gl.largeText)),
      ));
    }
    return items;
  }

  String getFirstLanguage() {
    return _languages[0];
  }
}
