# TranslationAppProject

<p>
        This tutorial gives readers a full understanding of seq2seq models and shows how to build a competitive seq2seq
        model from scratch. We focus on the task of Neural Machine Translation (NMT) which was the very first testbed
        for seq2seq models with wild success. The included code is lightweight, high-quality, production-ready, and
        incorporated with the latest research ideas.
 </p>
    <br>
    <p>
        Following the tutorial I created a neural machine translation for english to french, using encoder-decoder with standard attention.
        The encoder and decoder both had 4 layers of 1024 long short term memory units(LSTM), standard attention uses top layer of network
        to compute attention. 
        The data used is from <a href="http://www.statmt.org/wmt14/translation-task.html">Statistical Machine Translation</a>. My goal was
        to use the opensourced nmt code and create a api which would translate incoming sentences into the desired language, gaining a
        better understanding of RNNs, LSTM units, Attention mechanisms and Tensorflow. I trained the network for 2 days, by the 
        end it achieved a BLEU score of near 30. Note having been trained on text that is often news and political it fairs better at translating
        similar text, common vernacular will not translate well.
    </p>
    <br>
    <p>
        With the help of the following papers, I was able to gain a better understanding of the task at hand.
        <ul>
            <li>Sequence to Sequence Learning with Neural Networks(Ilya Sutskever, Oriol Vinyals, Quoc V. Le)</li>
            <li>Recurrent Neural Network Regularization(Wojciech Zaremba, Ilya Sutskever, Oriol Vinyals)</li>
            <li>Learning Phrase Representation using RNN Encoder-Decoder for Statistical Machine Translation(Kyunghyun Cho, Bart van Merriënboer, Et al.)</li>
            <li>Effective Approaches to Attention-based Neural Machine Translation(Minh-Thang Luong, Hieu Pham, Christopher D. Manning)</li>
        </ul>
    </p>
