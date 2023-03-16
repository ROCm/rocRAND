<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<tagfile>
  <compound kind="file">
    <name>hipranddevice.dox</name>
    <path>/home/samwu103/rocRAND/docs/</path>
    <filename>hipranddevice_8dox.html</filename>
  </compound>
  <compound kind="file">
    <name>hiprandhost.dox</name>
    <path>/home/samwu103/rocRAND/docs/</path>
    <filename>hiprandhost_8dox.html</filename>
  </compound>
  <compound kind="file">
    <name>hiprandhostcpp.dox</name>
    <path>/home/samwu103/rocRAND/docs/</path>
    <filename>hiprandhostcpp_8dox.html</filename>
  </compound>
  <compound kind="file">
    <name>mainpage.dox</name>
    <path>/home/samwu103/rocRAND/docs/</path>
    <filename>mainpage_8dox.html</filename>
  </compound>
  <compound kind="file">
    <name>rocranddevice.dox</name>
    <path>/home/samwu103/rocRAND/docs/</path>
    <filename>rocranddevice_8dox.html</filename>
  </compound>
  <compound kind="file">
    <name>rocrandhost.dox</name>
    <path>/home/samwu103/rocRAND/docs/</path>
    <filename>rocrandhost_8dox.html</filename>
  </compound>
  <compound kind="file">
    <name>rocrandhostcpp.dox</name>
    <path>/home/samwu103/rocRAND/docs/</path>
    <filename>rocrandhostcpp_8dox.html</filename>
  </compound>
  <compound kind="class">
    <name>rocrand_cpp::error</name>
    <filename>classrocrand__cpp_1_1error.html</filename>
    <member kind="typedef">
      <type>rocrand_status</type>
      <name>error_type</name>
      <anchorfile>classrocrand__cpp_1_1error.html</anchorfile>
      <anchor>acfa9d9b746ea314f330d83a9e86833c6</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>error</name>
      <anchorfile>classrocrand__cpp_1_1error.html</anchorfile>
      <anchor>aa831f47679c7f8deb8d50326862921fe</anchor>
      <arglist>(error_type error) noexcept</arglist>
    </member>
    <member kind="function">
      <type>error_type</type>
      <name>error_code</name>
      <anchorfile>classrocrand__cpp_1_1error.html</anchorfile>
      <anchor>a30ca1a12c8aeff5d6559f55d7e8cf58a</anchor>
      <arglist>() const noexcept</arglist>
    </member>
    <member kind="function">
      <type>std::string</type>
      <name>error_string</name>
      <anchorfile>classrocrand__cpp_1_1error.html</anchorfile>
      <anchor>a357342f9da29a4c278ec7639a5ecd2a5</anchor>
      <arglist>() const noexcept</arglist>
    </member>
    <member kind="function">
      <type>const char *</type>
      <name>what</name>
      <anchorfile>classrocrand__cpp_1_1error.html</anchorfile>
      <anchor>ae423d2235995eab71a5b7f3a3cde1f6a</anchor>
      <arglist>() const noexcept</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static std::string</type>
      <name>to_string</name>
      <anchorfile>classrocrand__cpp_1_1error.html</anchorfile>
      <anchor>aa01f72d0ff640df31a43882462f8e175</anchor>
      <arglist>(error_type error)</arglist>
    </member>
    <member kind="friend">
      <type>friend bool</type>
      <name>operator==</name>
      <anchorfile>classrocrand__cpp_1_1error.html</anchorfile>
      <anchor>ae9447c2d2b859fec52bd0a5556f41923</anchor>
      <arglist>(const error &amp;l, const error &amp;r)</arglist>
    </member>
    <member kind="friend">
      <type>friend bool</type>
      <name>operator!=</name>
      <anchorfile>classrocrand__cpp_1_1error.html</anchorfile>
      <anchor>a9c9588ec1a8faf308fd5c153445b43df</anchor>
      <arglist>(const error &amp;l, const error &amp;r)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rocrand_cpp::lfsr113_engine</name>
    <filename>classrocrand__cpp_1_1lfsr113__engine.html</filename>
    <templarg>DefaultSeedX</templarg>
    <templarg>DefaultSeedY</templarg>
    <templarg>DefaultSeedZ</templarg>
    <templarg>DefaultSeedW</templarg>
    <member kind="typedef">
      <type>unsigned int</type>
      <name>result_type</name>
      <anchorfile>classrocrand__cpp_1_1lfsr113__engine.html</anchorfile>
      <anchor>a7c0929b524367dce793310adbfc1f90c</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>rocrand_ordering</type>
      <name>order_type</name>
      <anchorfile>classrocrand__cpp_1_1lfsr113__engine.html</anchorfile>
      <anchor>add44d7a1c5a89ceccbec3ea9739304b6</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>uint4</type>
      <name>seed_type</name>
      <anchorfile>classrocrand__cpp_1_1lfsr113__engine.html</anchorfile>
      <anchor>a3865abf612e46310cf4c77662a75f056</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>lfsr113_engine</name>
      <anchorfile>classrocrand__cpp_1_1lfsr113__engine.html</anchorfile>
      <anchor>acaf1de4ddd550feecb6e6353b7d3903b</anchor>
      <arglist>(seed_type seed_value={DefaultSeedX, DefaultSeedY, DefaultSeedZ, DefaultSeedW}, order_type order_value=ROCRAND_ORDERING_QUASI_DEFAULT)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>lfsr113_engine</name>
      <anchorfile>classrocrand__cpp_1_1lfsr113__engine.html</anchorfile>
      <anchor>aaec55ca8140a8dd73d18632519bba12d</anchor>
      <arglist>(rocrand_generator &amp;generator)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>~lfsr113_engine</name>
      <anchorfile>classrocrand__cpp_1_1lfsr113__engine.html</anchorfile>
      <anchor>ae9e0fc1bb7ea3be1d5e1d83c2fcd687c</anchor>
      <arglist>() noexcept(false)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>stream</name>
      <anchorfile>classrocrand__cpp_1_1lfsr113__engine.html</anchorfile>
      <anchor>ab75ee9d5f820f97a502b3470b4d9a076</anchor>
      <arglist>(hipStream_t value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>order</name>
      <anchorfile>classrocrand__cpp_1_1lfsr113__engine.html</anchorfile>
      <anchor>ad00e2306d71b503b6d8fa83328557aa8</anchor>
      <arglist>(order_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>seed</name>
      <anchorfile>classrocrand__cpp_1_1lfsr113__engine.html</anchorfile>
      <anchor>a5a626e20b84c9c03e7812fc9ca1e8acd</anchor>
      <arglist>(unsigned long long value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>seed</name>
      <anchorfile>classrocrand__cpp_1_1lfsr113__engine.html</anchorfile>
      <anchor>a61dacdc0449eee5ff25756e2f7ab700a</anchor>
      <arglist>(seed_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>operator()</name>
      <anchorfile>classrocrand__cpp_1_1lfsr113__engine.html</anchorfile>
      <anchor>a0dbf74dac5b7abc262d1bf72450b8538</anchor>
      <arglist>(result_type *output, size_t size)</arglist>
    </member>
    <member kind="function">
      <type>result_type</type>
      <name>min</name>
      <anchorfile>classrocrand__cpp_1_1lfsr113__engine.html</anchorfile>
      <anchor>a328b2d94eef8ac97ad9fa6bd4d03a897</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>result_type</type>
      <name>max</name>
      <anchorfile>classrocrand__cpp_1_1lfsr113__engine.html</anchorfile>
      <anchor>ac09d63c3ad2319e726ed47251dab9dde</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static constexpr rocrand_rng_type</type>
      <name>type</name>
      <anchorfile>classrocrand__cpp_1_1lfsr113__engine.html</anchorfile>
      <anchor>a398d19e0b8004e65ca0e3b0febbef577</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="variable" static="yes">
      <type>static constexpr seed_type</type>
      <name>default_seed</name>
      <anchorfile>classrocrand__cpp_1_1lfsr113__engine.html</anchorfile>
      <anchor>a778b9191d41139f8d8183b1c0210b359</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rocrand_cpp::lognormal_distribution</name>
    <filename>classrocrand__cpp_1_1lognormal__distribution.html</filename>
    <templarg></templarg>
    <class kind="class">rocrand_cpp::lognormal_distribution::param_type</class>
    <member kind="function">
      <type></type>
      <name>lognormal_distribution</name>
      <anchorfile>classrocrand__cpp_1_1lognormal__distribution.html</anchorfile>
      <anchor>aaefb3fca639417ac60605d16d97b6405</anchor>
      <arglist>(RealType m=0.0, RealType s=1.0)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>lognormal_distribution</name>
      <anchorfile>classrocrand__cpp_1_1lognormal__distribution.html</anchorfile>
      <anchor>ae9e70d64bc474fec8843b7afe6e8c809</anchor>
      <arglist>(const param_type &amp;params)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>reset</name>
      <anchorfile>classrocrand__cpp_1_1lognormal__distribution.html</anchorfile>
      <anchor>a454d77063bb63a30115db44950c14671</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>RealType</type>
      <name>m</name>
      <anchorfile>classrocrand__cpp_1_1lognormal__distribution.html</anchorfile>
      <anchor>a900bf0f9cf95a5efc410b4625764bdda</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>RealType</type>
      <name>s</name>
      <anchorfile>classrocrand__cpp_1_1lognormal__distribution.html</anchorfile>
      <anchor>acde9976eed3742e994aa77af176d7804</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>param_type</type>
      <name>param</name>
      <anchorfile>classrocrand__cpp_1_1lognormal__distribution.html</anchorfile>
      <anchor>aecb67c6d357869fa630a378f6588506f</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>param</name>
      <anchorfile>classrocrand__cpp_1_1lognormal__distribution.html</anchorfile>
      <anchor>a0e6357318c9b75f6d1b8bdd911715520</anchor>
      <arglist>(const param_type &amp;params)</arglist>
    </member>
    <member kind="function">
      <type>RealType</type>
      <name>min</name>
      <anchorfile>classrocrand__cpp_1_1lognormal__distribution.html</anchorfile>
      <anchor>a594047e417fefe3341cb685b2bc14365</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>RealType</type>
      <name>max</name>
      <anchorfile>classrocrand__cpp_1_1lognormal__distribution.html</anchorfile>
      <anchor>a8052e36b5162c903cb707d1f9b493baa</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>operator()</name>
      <anchorfile>classrocrand__cpp_1_1lognormal__distribution.html</anchorfile>
      <anchor>a95c289fdb4ba2610dfecb0991efd5885</anchor>
      <arglist>(Generator &amp;g, RealType *output, size_t size)</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>operator==</name>
      <anchorfile>classrocrand__cpp_1_1lognormal__distribution.html</anchorfile>
      <anchor>ad48ae96ef66c2787b2e62a4229ed9008</anchor>
      <arglist>(const lognormal_distribution&lt; RealType &gt; &amp;other)</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>operator!=</name>
      <anchorfile>classrocrand__cpp_1_1lognormal__distribution.html</anchorfile>
      <anchor>afb3a41d017ee006ac333f5cb2173ea36</anchor>
      <arglist>(const lognormal_distribution&lt; RealType &gt; &amp;other)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rocrand_cpp::mrg31k3p_engine</name>
    <filename>classrocrand__cpp_1_1mrg31k3p__engine.html</filename>
    <templarg>DefaultSeed</templarg>
    <member kind="typedef">
      <type>unsigned int</type>
      <name>result_type</name>
      <anchorfile>classrocrand__cpp_1_1mrg31k3p__engine.html</anchorfile>
      <anchor>a3aa7792dbfb3479e1d7cf1a86177073c</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>rocrand_ordering</type>
      <name>order_type</name>
      <anchorfile>classrocrand__cpp_1_1mrg31k3p__engine.html</anchorfile>
      <anchor>a1dbd17082e90bf6419e797bdc3b4e355</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>unsigned long long</type>
      <name>offset_type</name>
      <anchorfile>classrocrand__cpp_1_1mrg31k3p__engine.html</anchorfile>
      <anchor>a38de74200b6527b8131e3458f3ae9188</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>unsigned long long</type>
      <name>seed_type</name>
      <anchorfile>classrocrand__cpp_1_1mrg31k3p__engine.html</anchorfile>
      <anchor>a31cea01eb233cbc34f2346cc7e5868ec</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>mrg31k3p_engine</name>
      <anchorfile>classrocrand__cpp_1_1mrg31k3p__engine.html</anchorfile>
      <anchor>a224ddee3ecb631c09b76945327bede46</anchor>
      <arglist>(seed_type seed_value=DefaultSeed, offset_type offset_value=0, order_type order_value=ROCRAND_ORDERING_PSEUDO_DEFAULT)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>mrg31k3p_engine</name>
      <anchorfile>classrocrand__cpp_1_1mrg31k3p__engine.html</anchorfile>
      <anchor>afb6f996a63624e5b82cfc92b86b8dcd9</anchor>
      <arglist>(rocrand_generator &amp;generator)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>~mrg31k3p_engine</name>
      <anchorfile>classrocrand__cpp_1_1mrg31k3p__engine.html</anchorfile>
      <anchor>a5686b7fcbb700296907a413d68dd1eba</anchor>
      <arglist>() noexcept(false)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>stream</name>
      <anchorfile>classrocrand__cpp_1_1mrg31k3p__engine.html</anchorfile>
      <anchor>a7232009709ccc1ca4fdb158e4b98cdc7</anchor>
      <arglist>(hipStream_t value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>order</name>
      <anchorfile>classrocrand__cpp_1_1mrg31k3p__engine.html</anchorfile>
      <anchor>a61f27942b076a6583836804e66245cc7</anchor>
      <arglist>(order_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>offset</name>
      <anchorfile>classrocrand__cpp_1_1mrg31k3p__engine.html</anchorfile>
      <anchor>a10987d357ae321ba3ba6fe8fe78838e8</anchor>
      <arglist>(offset_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>seed</name>
      <anchorfile>classrocrand__cpp_1_1mrg31k3p__engine.html</anchorfile>
      <anchor>afa015815c30a6409bd15c913ce320174</anchor>
      <arglist>(seed_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>operator()</name>
      <anchorfile>classrocrand__cpp_1_1mrg31k3p__engine.html</anchorfile>
      <anchor>a0a91ca511a63e342240ac738abf7b3e4</anchor>
      <arglist>(result_type *output, size_t size)</arglist>
    </member>
    <member kind="function">
      <type>result_type</type>
      <name>min</name>
      <anchorfile>classrocrand__cpp_1_1mrg31k3p__engine.html</anchorfile>
      <anchor>a35848a0c37fe2fca377580df4c8b9ea8</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>result_type</type>
      <name>max</name>
      <anchorfile>classrocrand__cpp_1_1mrg31k3p__engine.html</anchorfile>
      <anchor>a0e1bc75ba721bd6256124a4e9ce27032</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static constexpr rocrand_rng_type</type>
      <name>type</name>
      <anchorfile>classrocrand__cpp_1_1mrg31k3p__engine.html</anchorfile>
      <anchor>aeecad72012e1552c21d855780b79aa84</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="variable" static="yes">
      <type>static constexpr seed_type</type>
      <name>default_seed</name>
      <anchorfile>classrocrand__cpp_1_1mrg31k3p__engine.html</anchorfile>
      <anchor>a61fc847f3d5835343e50bfee60c371fb</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rocrand_cpp::mrg32k3a_engine</name>
    <filename>classrocrand__cpp_1_1mrg32k3a__engine.html</filename>
    <templarg>DefaultSeed</templarg>
    <member kind="typedef">
      <type>unsigned int</type>
      <name>result_type</name>
      <anchorfile>classrocrand__cpp_1_1mrg32k3a__engine.html</anchorfile>
      <anchor>a619aacffc7921fef8b9707b010c67278</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>rocrand_ordering</type>
      <name>order_type</name>
      <anchorfile>classrocrand__cpp_1_1mrg32k3a__engine.html</anchorfile>
      <anchor>a5b538a0cd95fe644fa4599f8688cd458</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>unsigned long long</type>
      <name>offset_type</name>
      <anchorfile>classrocrand__cpp_1_1mrg32k3a__engine.html</anchorfile>
      <anchor>a42745d3671719d9d549bfe8a5cf13183</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>unsigned long long</type>
      <name>seed_type</name>
      <anchorfile>classrocrand__cpp_1_1mrg32k3a__engine.html</anchorfile>
      <anchor>a3ec4de8c51d53d8d10d9acc598b39ca7</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>mrg32k3a_engine</name>
      <anchorfile>classrocrand__cpp_1_1mrg32k3a__engine.html</anchorfile>
      <anchor>ac9cb3b071944340be3cc28928511d692</anchor>
      <arglist>(seed_type seed_value=DefaultSeed, offset_type offset_value=0, order_type order_value=ROCRAND_ORDERING_PSEUDO_DEFAULT)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>mrg32k3a_engine</name>
      <anchorfile>classrocrand__cpp_1_1mrg32k3a__engine.html</anchorfile>
      <anchor>a94e4c3ca28587a0a50a52d3756c45527</anchor>
      <arglist>(rocrand_generator &amp;generator)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>~mrg32k3a_engine</name>
      <anchorfile>classrocrand__cpp_1_1mrg32k3a__engine.html</anchorfile>
      <anchor>a8ec9b0347f8fb2f8bac8b487f35b6063</anchor>
      <arglist>() noexcept(false)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>stream</name>
      <anchorfile>classrocrand__cpp_1_1mrg32k3a__engine.html</anchorfile>
      <anchor>ab594b1e4999bed8ae6bf4149745b2eda</anchor>
      <arglist>(hipStream_t value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>order</name>
      <anchorfile>classrocrand__cpp_1_1mrg32k3a__engine.html</anchorfile>
      <anchor>a7b4eb626a3c1218e8ea8bf4b16aa25a6</anchor>
      <arglist>(order_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>offset</name>
      <anchorfile>classrocrand__cpp_1_1mrg32k3a__engine.html</anchorfile>
      <anchor>a3134e7e74a917677d2c9fa8daaf13dfb</anchor>
      <arglist>(offset_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>seed</name>
      <anchorfile>classrocrand__cpp_1_1mrg32k3a__engine.html</anchorfile>
      <anchor>a69d54e2ca0509439f500839c1e8d1219</anchor>
      <arglist>(seed_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>operator()</name>
      <anchorfile>classrocrand__cpp_1_1mrg32k3a__engine.html</anchorfile>
      <anchor>ac3a814d0d242ec371fa57cb4f6f1916a</anchor>
      <arglist>(result_type *output, size_t size)</arglist>
    </member>
    <member kind="function">
      <type>result_type</type>
      <name>min</name>
      <anchorfile>classrocrand__cpp_1_1mrg32k3a__engine.html</anchorfile>
      <anchor>a1415821d352e6cae94cf2148ccdcc962</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>result_type</type>
      <name>max</name>
      <anchorfile>classrocrand__cpp_1_1mrg32k3a__engine.html</anchorfile>
      <anchor>a098f763f0d82a68e7a63cf28f97f39ee</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static constexpr rocrand_rng_type</type>
      <name>type</name>
      <anchorfile>classrocrand__cpp_1_1mrg32k3a__engine.html</anchorfile>
      <anchor>a30df6b1a96e62528d79ca5ff90ac3c78</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="variable" static="yes">
      <type>static constexpr seed_type</type>
      <name>default_seed</name>
      <anchorfile>classrocrand__cpp_1_1mrg32k3a__engine.html</anchorfile>
      <anchor>a8968a1f82defaf26eeaae16b17459b5d</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rocrand_cpp::mt19937_engine</name>
    <filename>classrocrand__cpp_1_1mt19937__engine.html</filename>
    <templarg>DefaultSeed</templarg>
    <member kind="typedef">
      <type>unsigned int</type>
      <name>result_type</name>
      <anchorfile>classrocrand__cpp_1_1mt19937__engine.html</anchorfile>
      <anchor>afce347e0d1e65002c70a93a3e942d31c</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>unsigned long long</type>
      <name>seed_type</name>
      <anchorfile>classrocrand__cpp_1_1mt19937__engine.html</anchorfile>
      <anchor>a2fd42aaefb5e532d9ef793a802006708</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>mt19937_engine</name>
      <anchorfile>classrocrand__cpp_1_1mt19937__engine.html</anchorfile>
      <anchor>af27effdddd38556a021e7ea99d31ac70</anchor>
      <arglist>(seed_type seed_value=DefaultSeed)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>mt19937_engine</name>
      <anchorfile>classrocrand__cpp_1_1mt19937__engine.html</anchorfile>
      <anchor>a9c8256e0130c9ff362a75a641d513c33</anchor>
      <arglist>(rocrand_generator &amp;generator)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>~mt19937_engine</name>
      <anchorfile>classrocrand__cpp_1_1mt19937__engine.html</anchorfile>
      <anchor>a1b9d3f1c5516355397115aafd0df9113</anchor>
      <arglist>() noexcept(false)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>stream</name>
      <anchorfile>classrocrand__cpp_1_1mt19937__engine.html</anchorfile>
      <anchor>a8f40adf074c9c1e0ee0d27b96aff0db6</anchor>
      <arglist>(hipStream_t value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>seed</name>
      <anchorfile>classrocrand__cpp_1_1mt19937__engine.html</anchorfile>
      <anchor>a1664c965a12a4d5450ecce80e3a76b2f</anchor>
      <arglist>(seed_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>operator()</name>
      <anchorfile>classrocrand__cpp_1_1mt19937__engine.html</anchorfile>
      <anchor>a8f2fb5e76ce1de6821798b2824bb3257</anchor>
      <arglist>(result_type *output, size_t size)</arglist>
    </member>
    <member kind="function">
      <type>result_type</type>
      <name>min</name>
      <anchorfile>classrocrand__cpp_1_1mt19937__engine.html</anchorfile>
      <anchor>aee514f5d39c0c9eb1d463b176abf055d</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>result_type</type>
      <name>max</name>
      <anchorfile>classrocrand__cpp_1_1mt19937__engine.html</anchorfile>
      <anchor>a0ac4e24b799e69f9c717be5a6fb8029f</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static constexpr rocrand_rng_type</type>
      <name>type</name>
      <anchorfile>classrocrand__cpp_1_1mt19937__engine.html</anchorfile>
      <anchor>ac57e1ec6ddcb076fbfd96cbf87d2eb84</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="variable" static="yes">
      <type>static constexpr seed_type</type>
      <name>default_seed</name>
      <anchorfile>classrocrand__cpp_1_1mt19937__engine.html</anchorfile>
      <anchor>aeafe28ce6a1dba9aa43a1d721b8dd0e7</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rocrand_cpp::mtgp32_engine</name>
    <filename>classrocrand__cpp_1_1mtgp32__engine.html</filename>
    <templarg>DefaultSeed</templarg>
    <member kind="typedef">
      <type>unsigned int</type>
      <name>result_type</name>
      <anchorfile>classrocrand__cpp_1_1mtgp32__engine.html</anchorfile>
      <anchor>a9e40ecb9236339a74ca62cc727fd42ee</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>rocrand_ordering</type>
      <name>order_type</name>
      <anchorfile>classrocrand__cpp_1_1mtgp32__engine.html</anchorfile>
      <anchor>af039621661f3d357ae976392cfc46a7f</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>unsigned long long</type>
      <name>offset_type</name>
      <anchorfile>classrocrand__cpp_1_1mtgp32__engine.html</anchorfile>
      <anchor>a87bec5463d173ebe0f7103a91c118176</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>unsigned long long</type>
      <name>seed_type</name>
      <anchorfile>classrocrand__cpp_1_1mtgp32__engine.html</anchorfile>
      <anchor>a46e872e5ff959e7fb34e6d4db38817c0</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>mtgp32_engine</name>
      <anchorfile>classrocrand__cpp_1_1mtgp32__engine.html</anchorfile>
      <anchor>a6de54be5367f4d9a9d03e8d17bbcbece</anchor>
      <arglist>(seed_type seed_value=DefaultSeed, order_type order_value=ROCRAND_ORDERING_PSEUDO_DEFAULT)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>mtgp32_engine</name>
      <anchorfile>classrocrand__cpp_1_1mtgp32__engine.html</anchorfile>
      <anchor>a544491c461e4195e121a0470ad1f2fae</anchor>
      <arglist>(rocrand_generator &amp;generator)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>~mtgp32_engine</name>
      <anchorfile>classrocrand__cpp_1_1mtgp32__engine.html</anchorfile>
      <anchor>a17abedea88faf7796d4abde23da2c7fc</anchor>
      <arglist>() noexcept(false)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>stream</name>
      <anchorfile>classrocrand__cpp_1_1mtgp32__engine.html</anchorfile>
      <anchor>a55d5657c858d9fe0a5ec56305f3ba350</anchor>
      <arglist>(hipStream_t value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>order</name>
      <anchorfile>classrocrand__cpp_1_1mtgp32__engine.html</anchorfile>
      <anchor>a71cb58322cb44a9859c35802f48ac6c1</anchor>
      <arglist>(order_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>seed</name>
      <anchorfile>classrocrand__cpp_1_1mtgp32__engine.html</anchorfile>
      <anchor>a226b0d69fc82ace309526399fe7b7544</anchor>
      <arglist>(seed_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>operator()</name>
      <anchorfile>classrocrand__cpp_1_1mtgp32__engine.html</anchorfile>
      <anchor>adb3a4752b39e9c67483190ea526a3a05</anchor>
      <arglist>(result_type *output, size_t size)</arglist>
    </member>
    <member kind="function">
      <type>result_type</type>
      <name>min</name>
      <anchorfile>classrocrand__cpp_1_1mtgp32__engine.html</anchorfile>
      <anchor>a0d4e635ce4686a47ad2a1d681e3b57d2</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>result_type</type>
      <name>max</name>
      <anchorfile>classrocrand__cpp_1_1mtgp32__engine.html</anchorfile>
      <anchor>a506955d38b326e9cca7a1a3bcb9bea14</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static constexpr rocrand_rng_type</type>
      <name>type</name>
      <anchorfile>classrocrand__cpp_1_1mtgp32__engine.html</anchorfile>
      <anchor>a3b66e36b635ae9f6a0ba02289f00b9aa</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="variable" static="yes">
      <type>static constexpr seed_type</type>
      <name>default_seed</name>
      <anchorfile>classrocrand__cpp_1_1mtgp32__engine.html</anchorfile>
      <anchor>a937c15a3de6c1b0ed1794337473090dd</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>mtgp32_params_fast_t</name>
    <filename>structmtgp32__params__fast__t.html</filename>
    <member kind="variable">
      <type>int</type>
      <name>mexp</name>
      <anchorfile>structmtgp32__params__fast__t.html</anchorfile>
      <anchor>a2c4be5532efdca6a9650c1460b20dcd7</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>pos</name>
      <anchorfile>structmtgp32__params__fast__t.html</anchorfile>
      <anchor>a25cdbfed1da9d2c429698f51f1cd1108</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>sh1</name>
      <anchorfile>structmtgp32__params__fast__t.html</anchorfile>
      <anchor>a7c30035713da02c6218f126875bcbaf0</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>sh2</name>
      <anchorfile>structmtgp32__params__fast__t.html</anchorfile>
      <anchor>a54e1d077b1a23eece387c89f4ce3604d</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>uint32_t</type>
      <name>tbl</name>
      <anchorfile>structmtgp32__params__fast__t.html</anchorfile>
      <anchor>a92d39b1a02d03834b74c47a68e80ce37</anchor>
      <arglist>[16]</arglist>
    </member>
    <member kind="variable">
      <type>uint32_t</type>
      <name>tmp_tbl</name>
      <anchorfile>structmtgp32__params__fast__t.html</anchorfile>
      <anchor>a03e17b91b1f02b7e337574fa797a40ee</anchor>
      <arglist>[16]</arglist>
    </member>
    <member kind="variable">
      <type>uint32_t</type>
      <name>flt_tmp_tbl</name>
      <anchorfile>structmtgp32__params__fast__t.html</anchorfile>
      <anchor>acf6032845969cd69195b797adeb4c901</anchor>
      <arglist>[16]</arglist>
    </member>
    <member kind="variable">
      <type>uint32_t</type>
      <name>mask</name>
      <anchorfile>structmtgp32__params__fast__t.html</anchorfile>
      <anchor>a8d630ac4c50518932098adc58eaead8a</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>unsigned char</type>
      <name>poly_sha1</name>
      <anchorfile>structmtgp32__params__fast__t.html</anchorfile>
      <anchor>a8e08f5b175887f09749145c724eb9bc2</anchor>
      <arglist>[21]</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rocrand_cpp::normal_distribution</name>
    <filename>classrocrand__cpp_1_1normal__distribution.html</filename>
    <templarg></templarg>
    <class kind="class">rocrand_cpp::normal_distribution::param_type</class>
    <member kind="function">
      <type></type>
      <name>normal_distribution</name>
      <anchorfile>classrocrand__cpp_1_1normal__distribution.html</anchorfile>
      <anchor>ad9b61030d49b8d1a1b5a1bc14ab963e1</anchor>
      <arglist>(RealType mean=0.0, RealType stddev=1.0)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>normal_distribution</name>
      <anchorfile>classrocrand__cpp_1_1normal__distribution.html</anchorfile>
      <anchor>a1603c2cc45645281a6c6094f66b1af85</anchor>
      <arglist>(const param_type &amp;params)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>reset</name>
      <anchorfile>classrocrand__cpp_1_1normal__distribution.html</anchorfile>
      <anchor>a84f7b0ddf61e04da4f741ad943efd5dc</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>RealType</type>
      <name>mean</name>
      <anchorfile>classrocrand__cpp_1_1normal__distribution.html</anchorfile>
      <anchor>a6938b6d53dae5bd2e3deff46337c413e</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>RealType</type>
      <name>stddev</name>
      <anchorfile>classrocrand__cpp_1_1normal__distribution.html</anchorfile>
      <anchor>ae622192633c60febc2592aed3a2bd836</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>RealType</type>
      <name>min</name>
      <anchorfile>classrocrand__cpp_1_1normal__distribution.html</anchorfile>
      <anchor>ab3dcf60ca987452460c005b9144dfea5</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>RealType</type>
      <name>max</name>
      <anchorfile>classrocrand__cpp_1_1normal__distribution.html</anchorfile>
      <anchor>a15f5f3cba194ea1f747d647e888e6eeb</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>param_type</type>
      <name>param</name>
      <anchorfile>classrocrand__cpp_1_1normal__distribution.html</anchorfile>
      <anchor>a624a351dd162874f375f228a6f294d0f</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>param</name>
      <anchorfile>classrocrand__cpp_1_1normal__distribution.html</anchorfile>
      <anchor>a2c11e9290b7a237c03dffdcdc001b5a2</anchor>
      <arglist>(const param_type &amp;params)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>operator()</name>
      <anchorfile>classrocrand__cpp_1_1normal__distribution.html</anchorfile>
      <anchor>a6bfdc37a97f8aab534f7c50fd15c6639</anchor>
      <arglist>(Generator &amp;g, RealType *output, size_t size)</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>operator==</name>
      <anchorfile>classrocrand__cpp_1_1normal__distribution.html</anchorfile>
      <anchor>a253a18d8c661f5d08947c31c66273f6a</anchor>
      <arglist>(const normal_distribution&lt; RealType &gt; &amp;other)</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>operator!=</name>
      <anchorfile>classrocrand__cpp_1_1normal__distribution.html</anchorfile>
      <anchor>adf5c3110f50b4d317c6bdddd4d81fc7f</anchor>
      <arglist>(const normal_distribution&lt; RealType &gt; &amp;other)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rocrand_cpp::poisson_distribution::param_type</name>
    <filename>classrocrand__cpp_1_1poisson__distribution_1_1param__type.html</filename>
    <member kind="function">
      <type>double</type>
      <name>mean</name>
      <anchorfile>classrocrand__cpp_1_1poisson__distribution_1_1param__type.html</anchorfile>
      <anchor>a68d7a61b52c340f8ded84f1efdffb9f8</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>operator==</name>
      <anchorfile>classrocrand__cpp_1_1poisson__distribution_1_1param__type.html</anchorfile>
      <anchor>ad6f7cbe4e375675d74c6a6b8af9b7c74</anchor>
      <arglist>(const param_type &amp;other)</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>operator!=</name>
      <anchorfile>classrocrand__cpp_1_1poisson__distribution_1_1param__type.html</anchorfile>
      <anchor>ac6f5f32a3ca0f579937799e25276fd65</anchor>
      <arglist>(const param_type &amp;other)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rocrand_cpp::normal_distribution::param_type</name>
    <filename>classrocrand__cpp_1_1normal__distribution_1_1param__type.html</filename>
    <member kind="function">
      <type>RealType</type>
      <name>mean</name>
      <anchorfile>classrocrand__cpp_1_1normal__distribution_1_1param__type.html</anchorfile>
      <anchor>a8a421257804aaafb4c08214b693d485e</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>RealType</type>
      <name>stddev</name>
      <anchorfile>classrocrand__cpp_1_1normal__distribution_1_1param__type.html</anchorfile>
      <anchor>ac6bb6f76bfcc7abbc78c8c714197faac</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>operator==</name>
      <anchorfile>classrocrand__cpp_1_1normal__distribution_1_1param__type.html</anchorfile>
      <anchor>a4ab4672860cded24df7a64fd718c0114</anchor>
      <arglist>(const param_type &amp;other)</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>operator!=</name>
      <anchorfile>classrocrand__cpp_1_1normal__distribution_1_1param__type.html</anchorfile>
      <anchor>aead99c8ca15fdddf08915b6f8bb85499</anchor>
      <arglist>(const param_type &amp;other)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rocrand_cpp::lognormal_distribution::param_type</name>
    <filename>classrocrand__cpp_1_1lognormal__distribution_1_1param__type.html</filename>
    <member kind="function">
      <type>RealType</type>
      <name>m</name>
      <anchorfile>classrocrand__cpp_1_1lognormal__distribution_1_1param__type.html</anchorfile>
      <anchor>a08ec3ccf9cdb24085d0284ebc1425f95</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>RealType</type>
      <name>s</name>
      <anchorfile>classrocrand__cpp_1_1lognormal__distribution_1_1param__type.html</anchorfile>
      <anchor>a662665935d57e09dd580b7b62aac04c5</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>operator==</name>
      <anchorfile>classrocrand__cpp_1_1lognormal__distribution_1_1param__type.html</anchorfile>
      <anchor>aae14bb51f809fc09310150da33909505</anchor>
      <arglist>(const param_type &amp;other)</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>operator!=</name>
      <anchorfile>classrocrand__cpp_1_1lognormal__distribution_1_1param__type.html</anchorfile>
      <anchor>a8abd7fbaa4d21e75ae3bc96d69b5d632</anchor>
      <arglist>(const param_type &amp;other)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rocrand_cpp::philox4x32_10_engine</name>
    <filename>classrocrand__cpp_1_1philox4x32__10__engine.html</filename>
    <templarg>DefaultSeed</templarg>
    <member kind="typedef">
      <type>unsigned int</type>
      <name>result_type</name>
      <anchorfile>classrocrand__cpp_1_1philox4x32__10__engine.html</anchorfile>
      <anchor>ad50648a958ff641a365740c5d12c3b32</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>rocrand_ordering</type>
      <name>order_type</name>
      <anchorfile>classrocrand__cpp_1_1philox4x32__10__engine.html</anchorfile>
      <anchor>aff68f0e0e2efae607814515de35c34f3</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>unsigned long long</type>
      <name>offset_type</name>
      <anchorfile>classrocrand__cpp_1_1philox4x32__10__engine.html</anchorfile>
      <anchor>a319f4c59827aca85830efda9df5ba88b</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>unsigned long long</type>
      <name>seed_type</name>
      <anchorfile>classrocrand__cpp_1_1philox4x32__10__engine.html</anchorfile>
      <anchor>a4ef6f7616a1c1e33f2640347d1deba85</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>philox4x32_10_engine</name>
      <anchorfile>classrocrand__cpp_1_1philox4x32__10__engine.html</anchorfile>
      <anchor>aca64651cbcd876c6fcaa19362ea03ce3</anchor>
      <arglist>(seed_type seed_value=DefaultSeed, offset_type offset_value=0, order_type order_value=ROCRAND_ORDERING_PSEUDO_DEFAULT)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>philox4x32_10_engine</name>
      <anchorfile>classrocrand__cpp_1_1philox4x32__10__engine.html</anchorfile>
      <anchor>a339933d59e534f95ac9b891fbacd9463</anchor>
      <arglist>(rocrand_generator &amp;generator)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>~philox4x32_10_engine</name>
      <anchorfile>classrocrand__cpp_1_1philox4x32__10__engine.html</anchorfile>
      <anchor>a7a1f9bb5c5595ca0f50d40ac49e479a6</anchor>
      <arglist>() noexcept(false)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>stream</name>
      <anchorfile>classrocrand__cpp_1_1philox4x32__10__engine.html</anchorfile>
      <anchor>a41628e245828d29ae8351c3d83fac93f</anchor>
      <arglist>(hipStream_t value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>order</name>
      <anchorfile>classrocrand__cpp_1_1philox4x32__10__engine.html</anchorfile>
      <anchor>ac5ca745fee6a9eec383fc5b1cc8ff5bf</anchor>
      <arglist>(order_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>offset</name>
      <anchorfile>classrocrand__cpp_1_1philox4x32__10__engine.html</anchorfile>
      <anchor>a2dcb9683a06aeaf8a5788540361b1312</anchor>
      <arglist>(offset_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>seed</name>
      <anchorfile>classrocrand__cpp_1_1philox4x32__10__engine.html</anchorfile>
      <anchor>a444f38707093aeac4c6c4648b8c7bc0c</anchor>
      <arglist>(seed_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>operator()</name>
      <anchorfile>classrocrand__cpp_1_1philox4x32__10__engine.html</anchorfile>
      <anchor>ac34669aeb1f206068a1bc42543eaae8e</anchor>
      <arglist>(result_type *output, size_t size)</arglist>
    </member>
    <member kind="function">
      <type>result_type</type>
      <name>min</name>
      <anchorfile>classrocrand__cpp_1_1philox4x32__10__engine.html</anchorfile>
      <anchor>ab29c6c9889ceb1eb186a4d5801f3d614</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>result_type</type>
      <name>max</name>
      <anchorfile>classrocrand__cpp_1_1philox4x32__10__engine.html</anchorfile>
      <anchor>a0c5f66504d83baa864ad42ffda266b9a</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static constexpr rocrand_rng_type</type>
      <name>type</name>
      <anchorfile>classrocrand__cpp_1_1philox4x32__10__engine.html</anchorfile>
      <anchor>a05d9ac171ff7088679db543e26eb0b81</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="variable" static="yes">
      <type>static constexpr seed_type</type>
      <name>default_seed</name>
      <anchorfile>classrocrand__cpp_1_1philox4x32__10__engine.html</anchorfile>
      <anchor>a12df42f9d43a7c34f36fd34717413961</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rocrand_cpp::poisson_distribution</name>
    <filename>classrocrand__cpp_1_1poisson__distribution.html</filename>
    <templarg></templarg>
    <class kind="class">rocrand_cpp::poisson_distribution::param_type</class>
    <member kind="function">
      <type></type>
      <name>poisson_distribution</name>
      <anchorfile>classrocrand__cpp_1_1poisson__distribution.html</anchorfile>
      <anchor>ab599c40bbe9ebcf28c686de99e37079f</anchor>
      <arglist>(double mean=1.0)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>poisson_distribution</name>
      <anchorfile>classrocrand__cpp_1_1poisson__distribution.html</anchorfile>
      <anchor>ab2a0ae997069b4ebad0769bac0c9abef</anchor>
      <arglist>(const param_type &amp;params)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>reset</name>
      <anchorfile>classrocrand__cpp_1_1poisson__distribution.html</anchorfile>
      <anchor>adb7c4be8dc2576cdeafd3f3ba074b468</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>double</type>
      <name>mean</name>
      <anchorfile>classrocrand__cpp_1_1poisson__distribution.html</anchorfile>
      <anchor>a0c620b7e5539794f6ab0485e17999952</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>IntType</type>
      <name>min</name>
      <anchorfile>classrocrand__cpp_1_1poisson__distribution.html</anchorfile>
      <anchor>afd2fac590d58e20d45f9a8ed257f6e4f</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>IntType</type>
      <name>max</name>
      <anchorfile>classrocrand__cpp_1_1poisson__distribution.html</anchorfile>
      <anchor>ab525153e8fa8cee7f11da5568481cedb</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>param_type</type>
      <name>param</name>
      <anchorfile>classrocrand__cpp_1_1poisson__distribution.html</anchorfile>
      <anchor>a9e0fd88df8334efab4e85baaafb50446</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>param</name>
      <anchorfile>classrocrand__cpp_1_1poisson__distribution.html</anchorfile>
      <anchor>ac19456540ac30d5a4e300cb0a83e4304</anchor>
      <arglist>(const param_type &amp;params)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>operator()</name>
      <anchorfile>classrocrand__cpp_1_1poisson__distribution.html</anchorfile>
      <anchor>aa582e915b4728b917bc75aac00e16ef9</anchor>
      <arglist>(Generator &amp;g, IntType *output, size_t size)</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>operator==</name>
      <anchorfile>classrocrand__cpp_1_1poisson__distribution.html</anchorfile>
      <anchor>a231d6c7a8f6f9fd95e49d7f3ba5a24e4</anchor>
      <arglist>(const poisson_distribution&lt; IntType &gt; &amp;other)</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>operator!=</name>
      <anchorfile>classrocrand__cpp_1_1poisson__distribution.html</anchorfile>
      <anchor>ae6645b686040c8c5d8d579ef258a2e88</anchor>
      <arglist>(const poisson_distribution&lt; IntType &gt; &amp;other)</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>rocrand_discrete_distribution_st</name>
    <filename>structrocrand__discrete__distribution__st.html</filename>
  </compound>
  <compound kind="class">
    <name>rocrand_cpp::scrambled_sobol32_engine</name>
    <filename>classrocrand__cpp_1_1scrambled__sobol32__engine.html</filename>
    <templarg>DefaultNumDimensions</templarg>
    <member kind="typedef">
      <type>unsigned int</type>
      <name>result_type</name>
      <anchorfile>classrocrand__cpp_1_1scrambled__sobol32__engine.html</anchorfile>
      <anchor>a1575276923b8093a87c5b5412adea763</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>unsigned long long</type>
      <name>offset_type</name>
      <anchorfile>classrocrand__cpp_1_1scrambled__sobol32__engine.html</anchorfile>
      <anchor>a5520a14cf4558ce2f5c69f482b1567bb</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>rocrand_ordering</type>
      <name>order_type</name>
      <anchorfile>classrocrand__cpp_1_1scrambled__sobol32__engine.html</anchorfile>
      <anchor>aa5ee68b09ab74f17dc17f2091e8fd778</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>unsigned int</type>
      <name>dimensions_num_type</name>
      <anchorfile>classrocrand__cpp_1_1scrambled__sobol32__engine.html</anchorfile>
      <anchor>a16527ecfd63e55d36cd35c81874b80b9</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>scrambled_sobol32_engine</name>
      <anchorfile>classrocrand__cpp_1_1scrambled__sobol32__engine.html</anchorfile>
      <anchor>a47d4896e663c9f0954221ef25fd26a49</anchor>
      <arglist>(dimensions_num_type num_of_dimensions=DefaultNumDimensions, offset_type offset_value=0, order_type order_value=ROCRAND_ORDERING_QUASI_DEFAULT)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>scrambled_sobol32_engine</name>
      <anchorfile>classrocrand__cpp_1_1scrambled__sobol32__engine.html</anchorfile>
      <anchor>a160f71010d3cd7303fcddafe247760e1</anchor>
      <arglist>(rocrand_generator &amp;generator)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>~scrambled_sobol32_engine</name>
      <anchorfile>classrocrand__cpp_1_1scrambled__sobol32__engine.html</anchorfile>
      <anchor>a77b2484635fbb59fe705317df28f42c6</anchor>
      <arglist>() noexcept(false)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>stream</name>
      <anchorfile>classrocrand__cpp_1_1scrambled__sobol32__engine.html</anchorfile>
      <anchor>a6be606adade854e5e6d7df6cd12b6582</anchor>
      <arglist>(hipStream_t value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>order</name>
      <anchorfile>classrocrand__cpp_1_1scrambled__sobol32__engine.html</anchorfile>
      <anchor>a204cb224067a5dc25d2eada115dd7414</anchor>
      <arglist>(order_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>offset</name>
      <anchorfile>classrocrand__cpp_1_1scrambled__sobol32__engine.html</anchorfile>
      <anchor>a7abe59091d9ecbac6ab67471ebe33e21</anchor>
      <arglist>(offset_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>dimensions</name>
      <anchorfile>classrocrand__cpp_1_1scrambled__sobol32__engine.html</anchorfile>
      <anchor>afd58cac260afc8b951b1f5cdd24d7717</anchor>
      <arglist>(dimensions_num_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>operator()</name>
      <anchorfile>classrocrand__cpp_1_1scrambled__sobol32__engine.html</anchorfile>
      <anchor>aee02416b247944d68227ffcc5430191e</anchor>
      <arglist>(result_type *output, size_t size)</arglist>
    </member>
    <member kind="function">
      <type>result_type</type>
      <name>min</name>
      <anchorfile>classrocrand__cpp_1_1scrambled__sobol32__engine.html</anchorfile>
      <anchor>a6e2eb3aa64e893b9e3b004489102ede1</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>result_type</type>
      <name>max</name>
      <anchorfile>classrocrand__cpp_1_1scrambled__sobol32__engine.html</anchorfile>
      <anchor>a320a9254d43587c0a596832a8abc7551</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static constexpr rocrand_rng_type</type>
      <name>type</name>
      <anchorfile>classrocrand__cpp_1_1scrambled__sobol32__engine.html</anchorfile>
      <anchor>ae81ae707fbfdddc778130c0edc41fab0</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="variable" static="yes">
      <type>static constexpr dimensions_num_type</type>
      <name>default_num_dimensions</name>
      <anchorfile>classrocrand__cpp_1_1scrambled__sobol32__engine.html</anchorfile>
      <anchor>aa183d264f2084d6eef83083f05f84838</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rocrand_cpp::scrambled_sobol64_engine</name>
    <filename>classrocrand__cpp_1_1scrambled__sobol64__engine.html</filename>
    <templarg>DefaultNumDimensions</templarg>
    <member kind="typedef">
      <type>unsigned long long int</type>
      <name>result_type</name>
      <anchorfile>classrocrand__cpp_1_1scrambled__sobol64__engine.html</anchorfile>
      <anchor>a48fc7bd3da802a3d2a9d967c5e455444</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>rocrand_ordering</type>
      <name>order_type</name>
      <anchorfile>classrocrand__cpp_1_1scrambled__sobol64__engine.html</anchorfile>
      <anchor>a7fae1400aa4de2ba7139bb505a230abc</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>unsigned long long int</type>
      <name>offset_type</name>
      <anchorfile>classrocrand__cpp_1_1scrambled__sobol64__engine.html</anchorfile>
      <anchor>a39db167a07b4d0979b2d4ece1fa2e43a</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>unsigned int</type>
      <name>dimensions_num_type</name>
      <anchorfile>classrocrand__cpp_1_1scrambled__sobol64__engine.html</anchorfile>
      <anchor>a124af781cfc6aafb5ea209ff24d100ac</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>scrambled_sobol64_engine</name>
      <anchorfile>classrocrand__cpp_1_1scrambled__sobol64__engine.html</anchorfile>
      <anchor>a2a8f3653f7cfef8e1db85c013dbed291</anchor>
      <arglist>(dimensions_num_type num_of_dimensions=DefaultNumDimensions, offset_type offset_value=0, order_type order_value=ROCRAND_ORDERING_QUASI_DEFAULT)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>scrambled_sobol64_engine</name>
      <anchorfile>classrocrand__cpp_1_1scrambled__sobol64__engine.html</anchorfile>
      <anchor>ae895ef1b24d98d376d831d03de61a6da</anchor>
      <arglist>(rocrand_generator &amp;generator)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>~scrambled_sobol64_engine</name>
      <anchorfile>classrocrand__cpp_1_1scrambled__sobol64__engine.html</anchorfile>
      <anchor>acb7422c5ed39c1bfd8b5acda8603c4a5</anchor>
      <arglist>() noexcept(false)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>stream</name>
      <anchorfile>classrocrand__cpp_1_1scrambled__sobol64__engine.html</anchorfile>
      <anchor>ac7bf17c0339ffa02519d3785a6b77e16</anchor>
      <arglist>(hipStream_t value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>order</name>
      <anchorfile>classrocrand__cpp_1_1scrambled__sobol64__engine.html</anchorfile>
      <anchor>ac47e69bd415313ee1fbef5db102e3474</anchor>
      <arglist>(order_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>offset</name>
      <anchorfile>classrocrand__cpp_1_1scrambled__sobol64__engine.html</anchorfile>
      <anchor>a697d4e03114737ef60eb10ad1bf1933d</anchor>
      <arglist>(offset_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>dimensions</name>
      <anchorfile>classrocrand__cpp_1_1scrambled__sobol64__engine.html</anchorfile>
      <anchor>a5225bcefe681e89db7e1aac48f2cfaca</anchor>
      <arglist>(dimensions_num_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>operator()</name>
      <anchorfile>classrocrand__cpp_1_1scrambled__sobol64__engine.html</anchorfile>
      <anchor>aa96190b2cfe929f9947de237b0db1d00</anchor>
      <arglist>(result_type *output, size_t size)</arglist>
    </member>
    <member kind="function">
      <type>result_type</type>
      <name>min</name>
      <anchorfile>classrocrand__cpp_1_1scrambled__sobol64__engine.html</anchorfile>
      <anchor>a860fc28e7b7c3d34bbcd4bfeb2d1e58b</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>result_type</type>
      <name>max</name>
      <anchorfile>classrocrand__cpp_1_1scrambled__sobol64__engine.html</anchorfile>
      <anchor>a7a4f7816def854596e2f024ad8fa741b</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static constexpr rocrand_rng_type</type>
      <name>type</name>
      <anchorfile>classrocrand__cpp_1_1scrambled__sobol64__engine.html</anchorfile>
      <anchor>afc1c5f47603862c170110f425c258c3d</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="variable" static="yes">
      <type>static constexpr dimensions_num_type</type>
      <name>default_num_dimensions</name>
      <anchorfile>classrocrand__cpp_1_1scrambled__sobol64__engine.html</anchorfile>
      <anchor>a77b052e927386181e94669aa93684eaf</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rocrand_cpp::sobol32_engine</name>
    <filename>classrocrand__cpp_1_1sobol32__engine.html</filename>
    <templarg>DefaultNumDimensions</templarg>
    <member kind="typedef">
      <type>unsigned int</type>
      <name>result_type</name>
      <anchorfile>classrocrand__cpp_1_1sobol32__engine.html</anchorfile>
      <anchor>a1c4c4befd4b081e7963193f1f2fc22dd</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>rocrand_ordering</type>
      <name>order_type</name>
      <anchorfile>classrocrand__cpp_1_1sobol32__engine.html</anchorfile>
      <anchor>ab8cb811cc4683fb03b05a5e2d18bb2e1</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>unsigned long long</type>
      <name>offset_type</name>
      <anchorfile>classrocrand__cpp_1_1sobol32__engine.html</anchorfile>
      <anchor>aa5b62c69d31f7e209413c121e23a3fc0</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>unsigned int</type>
      <name>dimensions_num_type</name>
      <anchorfile>classrocrand__cpp_1_1sobol32__engine.html</anchorfile>
      <anchor>aafb7c6e11ae2d8bf4341e6bbb9d60237</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>sobol32_engine</name>
      <anchorfile>classrocrand__cpp_1_1sobol32__engine.html</anchorfile>
      <anchor>a412c66bdb294ad5f5f60a3ae2ae4711e</anchor>
      <arglist>(dimensions_num_type num_of_dimensions=DefaultNumDimensions, offset_type offset_value=0, order_type order_value=ROCRAND_ORDERING_QUASI_DEFAULT)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>sobol32_engine</name>
      <anchorfile>classrocrand__cpp_1_1sobol32__engine.html</anchorfile>
      <anchor>a65a584117bf528cdf24acfb8fc1d2c63</anchor>
      <arglist>(rocrand_generator &amp;generator)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>~sobol32_engine</name>
      <anchorfile>classrocrand__cpp_1_1sobol32__engine.html</anchorfile>
      <anchor>a6fe98c5425c0728744f2399e41b2c2a2</anchor>
      <arglist>() noexcept(false)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>stream</name>
      <anchorfile>classrocrand__cpp_1_1sobol32__engine.html</anchorfile>
      <anchor>af9daadc409fb644f981f01dadc48b418</anchor>
      <arglist>(hipStream_t value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>order</name>
      <anchorfile>classrocrand__cpp_1_1sobol32__engine.html</anchorfile>
      <anchor>a7b06429c503e21c6e2df8302ba25380b</anchor>
      <arglist>(order_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>offset</name>
      <anchorfile>classrocrand__cpp_1_1sobol32__engine.html</anchorfile>
      <anchor>a7dab69fb17f8d9e08365c35eb9b51272</anchor>
      <arglist>(offset_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>dimensions</name>
      <anchorfile>classrocrand__cpp_1_1sobol32__engine.html</anchorfile>
      <anchor>a45f870a9dce16bfe53513437e1a45b2a</anchor>
      <arglist>(dimensions_num_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>operator()</name>
      <anchorfile>classrocrand__cpp_1_1sobol32__engine.html</anchorfile>
      <anchor>aa7b23fb2224926ff56a3a2b7a4b94601</anchor>
      <arglist>(result_type *output, size_t size)</arglist>
    </member>
    <member kind="function">
      <type>result_type</type>
      <name>min</name>
      <anchorfile>classrocrand__cpp_1_1sobol32__engine.html</anchorfile>
      <anchor>ae58a8cacf409aeb20d029652fcd5ea3a</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>result_type</type>
      <name>max</name>
      <anchorfile>classrocrand__cpp_1_1sobol32__engine.html</anchorfile>
      <anchor>af7bcc8332d285734bee621d417845092</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static constexpr rocrand_rng_type</type>
      <name>type</name>
      <anchorfile>classrocrand__cpp_1_1sobol32__engine.html</anchorfile>
      <anchor>ab70979330ea49835adcb62f64ffb9b20</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="variable" static="yes">
      <type>static constexpr dimensions_num_type</type>
      <name>default_num_dimensions</name>
      <anchorfile>classrocrand__cpp_1_1sobol32__engine.html</anchorfile>
      <anchor>ac9f4075aa758267d7bddaee1e2537f7d</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rocrand_cpp::sobol64_engine</name>
    <filename>classrocrand__cpp_1_1sobol64__engine.html</filename>
    <templarg>DefaultNumDimensions</templarg>
    <member kind="typedef">
      <type>unsigned long long int</type>
      <name>result_type</name>
      <anchorfile>classrocrand__cpp_1_1sobol64__engine.html</anchorfile>
      <anchor>a389b6e71f6a14ed76ec14b1a827030c3</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>unsigned long long int</type>
      <name>offset_type</name>
      <anchorfile>classrocrand__cpp_1_1sobol64__engine.html</anchorfile>
      <anchor>aa4eab606f20c8a313f9ac277bd0d52ba</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>rocrand_ordering</type>
      <name>order_type</name>
      <anchorfile>classrocrand__cpp_1_1sobol64__engine.html</anchorfile>
      <anchor>acba7c63748d7e454cb162ef5a3caf0c4</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>unsigned int</type>
      <name>dimensions_num_type</name>
      <anchorfile>classrocrand__cpp_1_1sobol64__engine.html</anchorfile>
      <anchor>a6f162ad16c20199f22fbbff7d8cc238f</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>sobol64_engine</name>
      <anchorfile>classrocrand__cpp_1_1sobol64__engine.html</anchorfile>
      <anchor>a6668a4ecdfefbd9086845235f520626c</anchor>
      <arglist>(dimensions_num_type num_of_dimensions=DefaultNumDimensions, offset_type offset_value=0, order_type order_value=ROCRAND_ORDERING_QUASI_DEFAULT)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>sobol64_engine</name>
      <anchorfile>classrocrand__cpp_1_1sobol64__engine.html</anchorfile>
      <anchor>ac83c0c49a136ddd2614682cc13af22d5</anchor>
      <arglist>(rocrand_generator &amp;generator)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>~sobol64_engine</name>
      <anchorfile>classrocrand__cpp_1_1sobol64__engine.html</anchorfile>
      <anchor>acd101586a1e763c63a3887e6b9e96a2c</anchor>
      <arglist>() noexcept(false)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>stream</name>
      <anchorfile>classrocrand__cpp_1_1sobol64__engine.html</anchorfile>
      <anchor>aa1ebaa92b5a5af0eb780e0ef897cd864</anchor>
      <arglist>(hipStream_t value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>order</name>
      <anchorfile>classrocrand__cpp_1_1sobol64__engine.html</anchorfile>
      <anchor>a2d1319c9c16819cedbd3d33b91c8c9ba</anchor>
      <arglist>(order_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>offset</name>
      <anchorfile>classrocrand__cpp_1_1sobol64__engine.html</anchorfile>
      <anchor>a12bf4224f56cb2b497fdca541c40be46</anchor>
      <arglist>(offset_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>dimensions</name>
      <anchorfile>classrocrand__cpp_1_1sobol64__engine.html</anchorfile>
      <anchor>a0828226d44e4c9a8a872805af390047b</anchor>
      <arglist>(dimensions_num_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>operator()</name>
      <anchorfile>classrocrand__cpp_1_1sobol64__engine.html</anchorfile>
      <anchor>ad2b213e62ffa4e5180906646233bd8da</anchor>
      <arglist>(result_type *output, size_t size)</arglist>
    </member>
    <member kind="function">
      <type>result_type</type>
      <name>min</name>
      <anchorfile>classrocrand__cpp_1_1sobol64__engine.html</anchorfile>
      <anchor>afe4486a603293f9218b587819e853b6f</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>result_type</type>
      <name>max</name>
      <anchorfile>classrocrand__cpp_1_1sobol64__engine.html</anchorfile>
      <anchor>af5b32270781607262476addac0e86a49</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static constexpr rocrand_rng_type</type>
      <name>type</name>
      <anchorfile>classrocrand__cpp_1_1sobol64__engine.html</anchorfile>
      <anchor>a478b0f63ac99a527a0f5f4115e58e057</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="variable" static="yes">
      <type>static constexpr dimensions_num_type</type>
      <name>default_num_dimensions</name>
      <anchorfile>classrocrand__cpp_1_1sobol64__engine.html</anchorfile>
      <anchor>a7a3156f799bd1718b99f824d5cfaa2c6</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rocrand_cpp::threefry2x32_20_engine</name>
    <filename>classrocrand__cpp_1_1threefry2x32__20__engine.html</filename>
    <templarg>DefaultSeed</templarg>
    <member kind="typedef">
      <type>unsigned int</type>
      <name>result_type</name>
      <anchorfile>classrocrand__cpp_1_1threefry2x32__20__engine.html</anchorfile>
      <anchor>a11dccd471b7e01f2f802a95b8ee3409b</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>rocrand_ordering</type>
      <name>order_type</name>
      <anchorfile>classrocrand__cpp_1_1threefry2x32__20__engine.html</anchorfile>
      <anchor>ad0bfdd0d7870d85045b50243c95aeb31</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>unsigned long long</type>
      <name>offset_type</name>
      <anchorfile>classrocrand__cpp_1_1threefry2x32__20__engine.html</anchorfile>
      <anchor>addf1855e1fcc7a3466b89e671ef0af82</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>unsigned long long</type>
      <name>seed_type</name>
      <anchorfile>classrocrand__cpp_1_1threefry2x32__20__engine.html</anchorfile>
      <anchor>a8e1be93926c51c706183436e38fe5f36</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>threefry2x32_20_engine</name>
      <anchorfile>classrocrand__cpp_1_1threefry2x32__20__engine.html</anchorfile>
      <anchor>a666ab718cacf4270cf281f4b490edcd1</anchor>
      <arglist>(seed_type seed_value=DefaultSeed, offset_type offset_value=0, order_type order_value=ROCRAND_ORDERING_PSEUDO_DEFAULT)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>threefry2x32_20_engine</name>
      <anchorfile>classrocrand__cpp_1_1threefry2x32__20__engine.html</anchorfile>
      <anchor>a3d9083e702691e931a3fa383fe71e551</anchor>
      <arglist>(rocrand_generator &amp;generator)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>~threefry2x32_20_engine</name>
      <anchorfile>classrocrand__cpp_1_1threefry2x32__20__engine.html</anchorfile>
      <anchor>a84d39c97de8d47e8e97151f5d2ade34d</anchor>
      <arglist>() noexcept(false)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>stream</name>
      <anchorfile>classrocrand__cpp_1_1threefry2x32__20__engine.html</anchorfile>
      <anchor>a1a17b771ce3926b9b11d0df822d17331</anchor>
      <arglist>(hipStream_t value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>order</name>
      <anchorfile>classrocrand__cpp_1_1threefry2x32__20__engine.html</anchorfile>
      <anchor>ae2abaa464f9f9cf233255b0d0e128615</anchor>
      <arglist>(order_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>offset</name>
      <anchorfile>classrocrand__cpp_1_1threefry2x32__20__engine.html</anchorfile>
      <anchor>a9ab0847352858e5504c4b09c0b9f61b6</anchor>
      <arglist>(offset_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>seed</name>
      <anchorfile>classrocrand__cpp_1_1threefry2x32__20__engine.html</anchorfile>
      <anchor>a98c3fafad63cec082ff826ab9cf17770</anchor>
      <arglist>(seed_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>operator()</name>
      <anchorfile>classrocrand__cpp_1_1threefry2x32__20__engine.html</anchorfile>
      <anchor>a946f7230b1edb71a0b3d2f9c322f14ff</anchor>
      <arglist>(result_type *output, size_t size)</arglist>
    </member>
    <member kind="function">
      <type>result_type</type>
      <name>min</name>
      <anchorfile>classrocrand__cpp_1_1threefry2x32__20__engine.html</anchorfile>
      <anchor>a0202608730dd0761fe85169dff655295</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>result_type</type>
      <name>max</name>
      <anchorfile>classrocrand__cpp_1_1threefry2x32__20__engine.html</anchorfile>
      <anchor>ae24be97b227f38239ecd5bd065a53fb1</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static constexpr rocrand_rng_type</type>
      <name>type</name>
      <anchorfile>classrocrand__cpp_1_1threefry2x32__20__engine.html</anchorfile>
      <anchor>a45d861d74fa3ce955b83a6667b9057cd</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="variable" static="yes">
      <type>static constexpr seed_type</type>
      <name>default_seed</name>
      <anchorfile>classrocrand__cpp_1_1threefry2x32__20__engine.html</anchorfile>
      <anchor>ab64762a727ac7701a7dbfcaf31b35dc3</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rocrand_cpp::threefry2x64_20_engine</name>
    <filename>classrocrand__cpp_1_1threefry2x64__20__engine.html</filename>
    <templarg>DefaultSeed</templarg>
    <member kind="typedef">
      <type>unsigned long long</type>
      <name>result_type</name>
      <anchorfile>classrocrand__cpp_1_1threefry2x64__20__engine.html</anchorfile>
      <anchor>a3437ddc7a586b6e42d84cd2d7b4338b9</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>rocrand_ordering</type>
      <name>order_type</name>
      <anchorfile>classrocrand__cpp_1_1threefry2x64__20__engine.html</anchorfile>
      <anchor>a7edd5cbb2724b2039eadbd932df81a26</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>unsigned long long</type>
      <name>offset_type</name>
      <anchorfile>classrocrand__cpp_1_1threefry2x64__20__engine.html</anchorfile>
      <anchor>acf55338cd83cbeaad24428f1d5b69797</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>unsigned long long</type>
      <name>seed_type</name>
      <anchorfile>classrocrand__cpp_1_1threefry2x64__20__engine.html</anchorfile>
      <anchor>a0c0d698444a37e1a1636569af1f25a03</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>threefry2x64_20_engine</name>
      <anchorfile>classrocrand__cpp_1_1threefry2x64__20__engine.html</anchorfile>
      <anchor>a7dd5cf6a389c66d2c2fbe0f19c842cde</anchor>
      <arglist>(seed_type seed_value=DefaultSeed, offset_type offset_value=0, order_type order_value=ROCRAND_ORDERING_PSEUDO_DEFAULT)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>threefry2x64_20_engine</name>
      <anchorfile>classrocrand__cpp_1_1threefry2x64__20__engine.html</anchorfile>
      <anchor>ad0b4170fb18673aed2e7380baf3ad44b</anchor>
      <arglist>(rocrand_generator &amp;generator)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>~threefry2x64_20_engine</name>
      <anchorfile>classrocrand__cpp_1_1threefry2x64__20__engine.html</anchorfile>
      <anchor>a6b40211e91511490c1281988af89be86</anchor>
      <arglist>() noexcept(false)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>stream</name>
      <anchorfile>classrocrand__cpp_1_1threefry2x64__20__engine.html</anchorfile>
      <anchor>ade41c644f8449b9c02ee7cad36d3e47a</anchor>
      <arglist>(hipStream_t value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>order</name>
      <anchorfile>classrocrand__cpp_1_1threefry2x64__20__engine.html</anchorfile>
      <anchor>ab649dd709b4a3e497f66fce401d33360</anchor>
      <arglist>(order_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>offset</name>
      <anchorfile>classrocrand__cpp_1_1threefry2x64__20__engine.html</anchorfile>
      <anchor>aa7755c220df71743a36eb2ad111a306c</anchor>
      <arglist>(offset_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>seed</name>
      <anchorfile>classrocrand__cpp_1_1threefry2x64__20__engine.html</anchorfile>
      <anchor>a43061a67ef31554b0182b22d3d993b23</anchor>
      <arglist>(seed_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>operator()</name>
      <anchorfile>classrocrand__cpp_1_1threefry2x64__20__engine.html</anchorfile>
      <anchor>aafca74edff8f732a101ba7b7870dd5d5</anchor>
      <arglist>(result_type *output, size_t size)</arglist>
    </member>
    <member kind="function">
      <type>result_type</type>
      <name>min</name>
      <anchorfile>classrocrand__cpp_1_1threefry2x64__20__engine.html</anchorfile>
      <anchor>ac3dc7bc71afb3e3e2fa89d9a13dd1879</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>result_type</type>
      <name>max</name>
      <anchorfile>classrocrand__cpp_1_1threefry2x64__20__engine.html</anchorfile>
      <anchor>ab5a43975f819fc8cf8eeb46f1cd984c7</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static constexpr rocrand_rng_type</type>
      <name>type</name>
      <anchorfile>classrocrand__cpp_1_1threefry2x64__20__engine.html</anchorfile>
      <anchor>a833b951fa297f68fa71913b45996641c</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="variable" static="yes">
      <type>static constexpr seed_type</type>
      <name>default_seed</name>
      <anchorfile>classrocrand__cpp_1_1threefry2x64__20__engine.html</anchorfile>
      <anchor>a1c38bbb4da54e23efe9ebced9fb4ceb8</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rocrand_cpp::threefry4x32_20_engine</name>
    <filename>classrocrand__cpp_1_1threefry4x32__20__engine.html</filename>
    <templarg>DefaultSeed</templarg>
    <member kind="typedef">
      <type>unsigned int</type>
      <name>result_type</name>
      <anchorfile>classrocrand__cpp_1_1threefry4x32__20__engine.html</anchorfile>
      <anchor>a754a127d78cbcb1bed271b3fce3b146f</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>rocrand_ordering</type>
      <name>order_type</name>
      <anchorfile>classrocrand__cpp_1_1threefry4x32__20__engine.html</anchorfile>
      <anchor>aaaa2f8b29e937164535d6c7fc5dd75a2</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>unsigned long long</type>
      <name>offset_type</name>
      <anchorfile>classrocrand__cpp_1_1threefry4x32__20__engine.html</anchorfile>
      <anchor>ac48fdb007d7fc5287345aa3d851f510a</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>unsigned long long</type>
      <name>seed_type</name>
      <anchorfile>classrocrand__cpp_1_1threefry4x32__20__engine.html</anchorfile>
      <anchor>a227f8c25af1ce955c685e12de05af53d</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>threefry4x32_20_engine</name>
      <anchorfile>classrocrand__cpp_1_1threefry4x32__20__engine.html</anchorfile>
      <anchor>a445782a0d9c3284f262bb32dcc299422</anchor>
      <arglist>(seed_type seed_value=DefaultSeed, offset_type offset_value=0, order_type order_value=ROCRAND_ORDERING_PSEUDO_DEFAULT)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>threefry4x32_20_engine</name>
      <anchorfile>classrocrand__cpp_1_1threefry4x32__20__engine.html</anchorfile>
      <anchor>ab9ebfabf73115cc38d7407580004bdca</anchor>
      <arglist>(rocrand_generator &amp;generator)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>~threefry4x32_20_engine</name>
      <anchorfile>classrocrand__cpp_1_1threefry4x32__20__engine.html</anchorfile>
      <anchor>a1bce3b1cf96b91b2bdebd817d19fc91a</anchor>
      <arglist>() noexcept(false)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>stream</name>
      <anchorfile>classrocrand__cpp_1_1threefry4x32__20__engine.html</anchorfile>
      <anchor>a6eb4fa5e0efd29f3ab09e693ea791c9c</anchor>
      <arglist>(hipStream_t value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>order</name>
      <anchorfile>classrocrand__cpp_1_1threefry4x32__20__engine.html</anchorfile>
      <anchor>aff89fbb2222f8a315926dad524bb1810</anchor>
      <arglist>(order_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>offset</name>
      <anchorfile>classrocrand__cpp_1_1threefry4x32__20__engine.html</anchorfile>
      <anchor>aa5d7649de54b122713cf2b68a29844f3</anchor>
      <arglist>(offset_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>seed</name>
      <anchorfile>classrocrand__cpp_1_1threefry4x32__20__engine.html</anchorfile>
      <anchor>aa7d50f9810649d393e62031cad823e94</anchor>
      <arglist>(seed_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>operator()</name>
      <anchorfile>classrocrand__cpp_1_1threefry4x32__20__engine.html</anchorfile>
      <anchor>ac5074caa9d7ec6f470f9fc5a247a994e</anchor>
      <arglist>(result_type *output, size_t size)</arglist>
    </member>
    <member kind="function">
      <type>result_type</type>
      <name>min</name>
      <anchorfile>classrocrand__cpp_1_1threefry4x32__20__engine.html</anchorfile>
      <anchor>a856fd7d6e76c943f5cdd49ee97a60404</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>result_type</type>
      <name>max</name>
      <anchorfile>classrocrand__cpp_1_1threefry4x32__20__engine.html</anchorfile>
      <anchor>a3adedd51d4e0a5d3a0df2967e4173259</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static constexpr rocrand_rng_type</type>
      <name>type</name>
      <anchorfile>classrocrand__cpp_1_1threefry4x32__20__engine.html</anchorfile>
      <anchor>ad2f36bcc1d2ac806e674e5efaf5e96bb</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="variable" static="yes">
      <type>static constexpr seed_type</type>
      <name>default_seed</name>
      <anchorfile>classrocrand__cpp_1_1threefry4x32__20__engine.html</anchorfile>
      <anchor>adb3ee4a59a7c97701ee39893cd87b320</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rocrand_cpp::threefry4x64_20_engine</name>
    <filename>classrocrand__cpp_1_1threefry4x64__20__engine.html</filename>
    <templarg>DefaultSeed</templarg>
    <member kind="typedef">
      <type>unsigned long long</type>
      <name>result_type</name>
      <anchorfile>classrocrand__cpp_1_1threefry4x64__20__engine.html</anchorfile>
      <anchor>ae3584e3d5df8219e948d285d64e35354</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>rocrand_ordering</type>
      <name>order_type</name>
      <anchorfile>classrocrand__cpp_1_1threefry4x64__20__engine.html</anchorfile>
      <anchor>af86ec5e8f95ddb714111ad30c9424320</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>unsigned long long</type>
      <name>offset_type</name>
      <anchorfile>classrocrand__cpp_1_1threefry4x64__20__engine.html</anchorfile>
      <anchor>a7e55cd26d72fa9756a767fb0565edbb2</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>unsigned long long</type>
      <name>seed_type</name>
      <anchorfile>classrocrand__cpp_1_1threefry4x64__20__engine.html</anchorfile>
      <anchor>a4306b00a8768ea35805f3adb9ef2bc6d</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>threefry4x64_20_engine</name>
      <anchorfile>classrocrand__cpp_1_1threefry4x64__20__engine.html</anchorfile>
      <anchor>a1dacbc9d623925eea4f00350253af56e</anchor>
      <arglist>(seed_type seed_value=DefaultSeed, offset_type offset_value=0, order_type order_value=ROCRAND_ORDERING_PSEUDO_DEFAULT)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>threefry4x64_20_engine</name>
      <anchorfile>classrocrand__cpp_1_1threefry4x64__20__engine.html</anchorfile>
      <anchor>a32c83f18fe154a894a2d900bdddabb14</anchor>
      <arglist>(rocrand_generator &amp;generator)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>~threefry4x64_20_engine</name>
      <anchorfile>classrocrand__cpp_1_1threefry4x64__20__engine.html</anchorfile>
      <anchor>a463a026bd303eaebc7fd37e95234bff6</anchor>
      <arglist>() noexcept(false)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>stream</name>
      <anchorfile>classrocrand__cpp_1_1threefry4x64__20__engine.html</anchorfile>
      <anchor>a5e93ed0b27508fc6a0779f96f8201930</anchor>
      <arglist>(hipStream_t value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>offset</name>
      <anchorfile>classrocrand__cpp_1_1threefry4x64__20__engine.html</anchorfile>
      <anchor>a4544be0ebb37c0f3f74920b57856094f</anchor>
      <arglist>(offset_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>seed</name>
      <anchorfile>classrocrand__cpp_1_1threefry4x64__20__engine.html</anchorfile>
      <anchor>afbb2aa87498d489d444e1d3c2f72e764</anchor>
      <arglist>(seed_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>operator()</name>
      <anchorfile>classrocrand__cpp_1_1threefry4x64__20__engine.html</anchorfile>
      <anchor>a954cac4182b57e9f63c704618f48abc7</anchor>
      <arglist>(result_type *output, size_t size)</arglist>
    </member>
    <member kind="function">
      <type>result_type</type>
      <name>min</name>
      <anchorfile>classrocrand__cpp_1_1threefry4x64__20__engine.html</anchorfile>
      <anchor>a6bf1787aa43d32e10350c38b0e5604ac</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>result_type</type>
      <name>max</name>
      <anchorfile>classrocrand__cpp_1_1threefry4x64__20__engine.html</anchorfile>
      <anchor>abecb70640d0e2e48f6f0c648d0f7682e</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static constexpr rocrand_rng_type</type>
      <name>type</name>
      <anchorfile>classrocrand__cpp_1_1threefry4x64__20__engine.html</anchorfile>
      <anchor>a5d7fd927212dff62a3ef274d836a35d7</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="variable" static="yes">
      <type>static constexpr seed_type</type>
      <name>default_seed</name>
      <anchorfile>classrocrand__cpp_1_1threefry4x64__20__engine.html</anchorfile>
      <anchor>a3e1e6ea634df52ef41132309a78ebe6a</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rocrand_cpp::uniform_int_distribution</name>
    <filename>classrocrand__cpp_1_1uniform__int__distribution.html</filename>
    <templarg></templarg>
    <member kind="function">
      <type></type>
      <name>uniform_int_distribution</name>
      <anchorfile>classrocrand__cpp_1_1uniform__int__distribution.html</anchorfile>
      <anchor>aa922c9ab5364cc066e5737c19e7c1b40</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>reset</name>
      <anchorfile>classrocrand__cpp_1_1uniform__int__distribution.html</anchorfile>
      <anchor>a3fd0e84f61a80549691d9ef1482a3f0f</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>IntType</type>
      <name>min</name>
      <anchorfile>classrocrand__cpp_1_1uniform__int__distribution.html</anchorfile>
      <anchor>a38d5c336fd0aa176a964252c83598fc7</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>IntType</type>
      <name>max</name>
      <anchorfile>classrocrand__cpp_1_1uniform__int__distribution.html</anchorfile>
      <anchor>a0cdb7407444fe476a0f5b2f62af8214b</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>operator()</name>
      <anchorfile>classrocrand__cpp_1_1uniform__int__distribution.html</anchorfile>
      <anchor>a5f0668243f0e24b46917e1207a111a88</anchor>
      <arglist>(Generator &amp;g, IntType *output, size_t size)</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>operator==</name>
      <anchorfile>classrocrand__cpp_1_1uniform__int__distribution.html</anchorfile>
      <anchor>af7c2988d3eecfe91ad8a713ac828a43a</anchor>
      <arglist>(const uniform_int_distribution&lt; IntType &gt; &amp;other)</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>operator!=</name>
      <anchorfile>classrocrand__cpp_1_1uniform__int__distribution.html</anchorfile>
      <anchor>affb542d563cfe73a74294c48854e6056</anchor>
      <arglist>(const uniform_int_distribution&lt; IntType &gt; &amp;other)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rocrand_cpp::uniform_real_distribution</name>
    <filename>classrocrand__cpp_1_1uniform__real__distribution.html</filename>
    <templarg></templarg>
    <member kind="function">
      <type></type>
      <name>uniform_real_distribution</name>
      <anchorfile>classrocrand__cpp_1_1uniform__real__distribution.html</anchorfile>
      <anchor>a5d2d39c6eca55aff2e0737735ddd86b8</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>reset</name>
      <anchorfile>classrocrand__cpp_1_1uniform__real__distribution.html</anchorfile>
      <anchor>a154e0b4d5753ec8b47ce1b3ff0cd4262</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>RealType</type>
      <name>min</name>
      <anchorfile>classrocrand__cpp_1_1uniform__real__distribution.html</anchorfile>
      <anchor>a0b8214e5e95a83baa4f4f57a821c9c83</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>RealType</type>
      <name>max</name>
      <anchorfile>classrocrand__cpp_1_1uniform__real__distribution.html</anchorfile>
      <anchor>a5398453ddde83e5cadd700d15e526bc7</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>operator()</name>
      <anchorfile>classrocrand__cpp_1_1uniform__real__distribution.html</anchorfile>
      <anchor>a9d88cf1e9d15f69ecfee0323b8ee4b4c</anchor>
      <arglist>(Generator &amp;g, RealType *output, size_t size)</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>operator==</name>
      <anchorfile>classrocrand__cpp_1_1uniform__real__distribution.html</anchorfile>
      <anchor>a3ebd0aa07cd9cf9df51d9acdd7f5d545</anchor>
      <arglist>(const uniform_real_distribution&lt; RealType &gt; &amp;other)</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>operator!=</name>
      <anchorfile>classrocrand__cpp_1_1uniform__real__distribution.html</anchorfile>
      <anchor>aee3af4689592b443f1b407271b03c9e2</anchor>
      <arglist>(const uniform_real_distribution&lt; RealType &gt; &amp;other)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>rocrand_cpp::xorwow_engine</name>
    <filename>classrocrand__cpp_1_1xorwow__engine.html</filename>
    <templarg>DefaultSeed</templarg>
    <member kind="typedef">
      <type>unsigned int</type>
      <name>result_type</name>
      <anchorfile>classrocrand__cpp_1_1xorwow__engine.html</anchorfile>
      <anchor>ac0fcffa76e704b5b9df76e8fc379733f</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>rocrand_ordering</type>
      <name>order_type</name>
      <anchorfile>classrocrand__cpp_1_1xorwow__engine.html</anchorfile>
      <anchor>a522de59ef540bfb20be6e7d3deb1ff51</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>unsigned long long</type>
      <name>offset_type</name>
      <anchorfile>classrocrand__cpp_1_1xorwow__engine.html</anchorfile>
      <anchor>ae3af2b1fa581be95da7b9300e936c436</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>unsigned long long</type>
      <name>seed_type</name>
      <anchorfile>classrocrand__cpp_1_1xorwow__engine.html</anchorfile>
      <anchor>ae10913dba532d4a6d6c1be095abdc732</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>xorwow_engine</name>
      <anchorfile>classrocrand__cpp_1_1xorwow__engine.html</anchorfile>
      <anchor>a7abd2f511034a84ef83d1054b1c71866</anchor>
      <arglist>(seed_type seed_value=DefaultSeed, offset_type offset_value=0, order_type order_value=ROCRAND_ORDERING_PSEUDO_DEFAULT)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>xorwow_engine</name>
      <anchorfile>classrocrand__cpp_1_1xorwow__engine.html</anchorfile>
      <anchor>a925d3e0dbbdb36564a28c1b50950aa44</anchor>
      <arglist>(rocrand_generator &amp;generator)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>~xorwow_engine</name>
      <anchorfile>classrocrand__cpp_1_1xorwow__engine.html</anchorfile>
      <anchor>aa6f934003a3e901148614a11b72ec864</anchor>
      <arglist>() noexcept(false)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>stream</name>
      <anchorfile>classrocrand__cpp_1_1xorwow__engine.html</anchorfile>
      <anchor>a10fb8b291bb339843925f4e882959399</anchor>
      <arglist>(hipStream_t value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>order</name>
      <anchorfile>classrocrand__cpp_1_1xorwow__engine.html</anchorfile>
      <anchor>a10f122de1e79a551622064ecedf8614d</anchor>
      <arglist>(order_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>offset</name>
      <anchorfile>classrocrand__cpp_1_1xorwow__engine.html</anchorfile>
      <anchor>aab83a280403c73be8a6175f16ef5598c</anchor>
      <arglist>(offset_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>seed</name>
      <anchorfile>classrocrand__cpp_1_1xorwow__engine.html</anchorfile>
      <anchor>a3eb344aa2025b32b40f84c6b440ab025</anchor>
      <arglist>(seed_type value)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>operator()</name>
      <anchorfile>classrocrand__cpp_1_1xorwow__engine.html</anchorfile>
      <anchor>aa917f2bd34ca6beb4c9c61cd93e06814</anchor>
      <arglist>(result_type *output, size_t size)</arglist>
    </member>
    <member kind="function">
      <type>result_type</type>
      <name>min</name>
      <anchorfile>classrocrand__cpp_1_1xorwow__engine.html</anchorfile>
      <anchor>a373d25b6a271787e66ce360451bfc783</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>result_type</type>
      <name>max</name>
      <anchorfile>classrocrand__cpp_1_1xorwow__engine.html</anchorfile>
      <anchor>aa9d9848d353d44dbd4d22aefe127f3a3</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function" static="yes">
      <type>static constexpr rocrand_rng_type</type>
      <name>type</name>
      <anchorfile>classrocrand__cpp_1_1xorwow__engine.html</anchorfile>
      <anchor>a34e0700397430216d69f078c7ec32369</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="variable" static="yes">
      <type>static constexpr seed_type</type>
      <name>default_seed</name>
      <anchorfile>classrocrand__cpp_1_1xorwow__engine.html</anchorfile>
      <anchor>acc333dc181a70af7b5572988d7c9a2ac</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="group">
    <name>rocrandhost</name>
    <title>rocRAND host API</title>
    <filename>group__rocrandhost.html</filename>
    <member kind="typedef">
      <type>enum rocrand_status</type>
      <name>rocrand_status</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gadf01dd2ebdbf98fd0165ac427812ea7b</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>enum rocrand_rng_type</type>
      <name>rocrand_rng_type</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>ga4f3010137ac3fba544d5b81d0febb76a</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>enum rocrand_ordering</type>
      <name>rocrand_ordering</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gad6b0e885ebdfbcbd0e650b4f5e9fbcfd</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumeration">
      <type></type>
      <name>rocrand_status</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>ga8baefa0d48532cd7d239f0f9cf8a36b7</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>ROCRAND_STATUS_SUCCESS</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gga8baefa0d48532cd7d239f0f9cf8a36b7a5637c668e1f344bdd9b03d237ef5bf7f</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>ROCRAND_STATUS_VERSION_MISMATCH</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gga8baefa0d48532cd7d239f0f9cf8a36b7a27406c4e369652c161c3846c975d4212</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>ROCRAND_STATUS_NOT_CREATED</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gga8baefa0d48532cd7d239f0f9cf8a36b7a11b69d6b4f12084309743dbf410e1bba</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>ROCRAND_STATUS_ALLOCATION_FAILED</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gga8baefa0d48532cd7d239f0f9cf8a36b7ae857b00237b8464d1e92afc69db79746</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>ROCRAND_STATUS_TYPE_ERROR</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gga8baefa0d48532cd7d239f0f9cf8a36b7af0ff3448f279a13c7a7a6f9b326317d4</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>ROCRAND_STATUS_OUT_OF_RANGE</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gga8baefa0d48532cd7d239f0f9cf8a36b7a8e77c368f38af31378bbe396af85071c</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>ROCRAND_STATUS_LENGTH_NOT_MULTIPLE</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gga8baefa0d48532cd7d239f0f9cf8a36b7a33e647d9cebaaea10272be5a3389ee41</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gga8baefa0d48532cd7d239f0f9cf8a36b7a4624012d6838616cad7e5759b0c99a18</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>ROCRAND_STATUS_LAUNCH_FAILURE</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gga8baefa0d48532cd7d239f0f9cf8a36b7a1daefd533a02ef65b2220ba67c0a98ef</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>ROCRAND_STATUS_INTERNAL_ERROR</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gga8baefa0d48532cd7d239f0f9cf8a36b7a5b5f6efbea8305bf1f4b2dd6986347b4</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumeration">
      <type></type>
      <name>rocrand_rng_type</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>ga993ef21b35670ba85459f27e5ceff63c</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>ROCRAND_RNG_PSEUDO_DEFAULT</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gga993ef21b35670ba85459f27e5ceff63ca0e3ec56247b4ebf27c868dd8e7e7c975</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>ROCRAND_RNG_PSEUDO_XORWOW</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gga993ef21b35670ba85459f27e5ceff63ca4c69983abe43422dd63ded337838fc86</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>ROCRAND_RNG_PSEUDO_MRG32K3A</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gga993ef21b35670ba85459f27e5ceff63caf9e08b618f5be8d60ebeeb609cba4d78</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>ROCRAND_RNG_PSEUDO_MTGP32</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gga993ef21b35670ba85459f27e5ceff63ca70542ed568393b589c4bdb0e095c217a</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>ROCRAND_RNG_PSEUDO_PHILOX4_32_10</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gga993ef21b35670ba85459f27e5ceff63ca63788724014a57b525d0956db01253e3</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>ROCRAND_RNG_PSEUDO_MRG31K3P</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gga993ef21b35670ba85459f27e5ceff63ca55ec2966fe80e3f7e345dc798e5f1f17</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>ROCRAND_RNG_PSEUDO_LFSR113</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gga993ef21b35670ba85459f27e5ceff63ca30f00beffb679f08828ffb1fe3dbe18a</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>ROCRAND_RNG_PSEUDO_MT19937</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gga993ef21b35670ba85459f27e5ceff63ca62038fa66fd37cef8c18c83a086285a6</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>ROCRAND_RNG_PSEUDO_THREEFRY2_32_20</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gga993ef21b35670ba85459f27e5ceff63ca63f573f6d6353009b8ffdcd7304960d8</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>ROCRAND_RNG_PSEUDO_THREEFRY2_64_20</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gga993ef21b35670ba85459f27e5ceff63ca3b06b10054af354b987304dbe31ef331</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>ROCRAND_RNG_PSEUDO_THREEFRY4_32_20</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gga993ef21b35670ba85459f27e5ceff63ca4da3806708ff9853c1fca64da83a6283</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>ROCRAND_RNG_PSEUDO_THREEFRY4_64_20</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gga993ef21b35670ba85459f27e5ceff63ca663c6fec4d8fb318e07c78cab3894216</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>ROCRAND_RNG_QUASI_DEFAULT</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gga993ef21b35670ba85459f27e5ceff63ca411554d2d63dc9b0061b520641f70138</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>ROCRAND_RNG_QUASI_SOBOL32</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gga993ef21b35670ba85459f27e5ceff63ca761e3aaea6e175f4f40880b3b3764f88</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gga993ef21b35670ba85459f27e5ceff63caa1b0a46966437ac6e1e58bf6ff740f91</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>ROCRAND_RNG_QUASI_SOBOL64</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gga993ef21b35670ba85459f27e5ceff63ca2380d4b5c804e9489e9515c8838e9e3e</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gga993ef21b35670ba85459f27e5ceff63caac11ede9f85266b48624ce1077cdc9dd</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumeration">
      <type></type>
      <name>rocrand_ordering</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>ga7e83887c5123bcb0778bafb0bcc45d03</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>ROCRAND_ORDERING_PSEUDO_BEST</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gga7e83887c5123bcb0778bafb0bcc45d03ae24279f855adacac25bcd99e4f9cbe92</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>ROCRAND_ORDERING_PSEUDO_DEFAULT</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gga7e83887c5123bcb0778bafb0bcc45d03a0491b4c1db82216f0e17ca62a37e7c5f</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>ROCRAND_ORDERING_PSEUDO_SEEDED</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gga7e83887c5123bcb0778bafb0bcc45d03a8b3d8fac9a7593ddcb0d8168cf9b9c35</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>ROCRAND_ORDERING_PSEUDO_LEGACY</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gga7e83887c5123bcb0778bafb0bcc45d03ae9496a66a7370b6396aadb1942d18d90</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>ROCRAND_ORDERING_PSEUDO_DYNAMIC</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gga7e83887c5123bcb0778bafb0bcc45d03ad3c0525d92a4d30eb36fbb0a02893b7c</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumvalue">
      <name>ROCRAND_ORDERING_QUASI_DEFAULT</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gga7e83887c5123bcb0778bafb0bcc45d03a4519701009c5801da1073b1b435090f8</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type>rocrand_status ROCRANDAPI</type>
      <name>rocrand_create_generator</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>ga8d0449c097371bbcbd28addd1e7f77b9</anchor>
      <arglist>(rocrand_generator *generator, rocrand_rng_type rng_type)</arglist>
    </member>
    <member kind="function">
      <type>rocrand_status ROCRANDAPI</type>
      <name>rocrand_destroy_generator</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gafc4dd689a2b5144aa6a788a33d0e32c2</anchor>
      <arglist>(rocrand_generator generator)</arglist>
    </member>
    <member kind="function">
      <type>rocrand_status ROCRANDAPI</type>
      <name>rocrand_generate</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gae3e149217c6ad892e2c46c7122756099</anchor>
      <arglist>(rocrand_generator generator, unsigned int *output_data, size_t n)</arglist>
    </member>
    <member kind="function">
      <type>rocrand_status ROCRANDAPI</type>
      <name>rocrand_generate_long_long</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>ga9e095f3670624a2312c1f19bc235b939</anchor>
      <arglist>(rocrand_generator generator, unsigned long long int *output_data, size_t n)</arglist>
    </member>
    <member kind="function">
      <type>rocrand_status ROCRANDAPI</type>
      <name>rocrand_generate_char</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>ga803fa46f877d00a58a02816b35f28b5d</anchor>
      <arglist>(rocrand_generator generator, unsigned char *output_data, size_t n)</arglist>
    </member>
    <member kind="function">
      <type>rocrand_status ROCRANDAPI</type>
      <name>rocrand_generate_short</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>ga062bd3bc4187a0ee9860fdbeda6cb3e3</anchor>
      <arglist>(rocrand_generator generator, unsigned short *output_data, size_t n)</arglist>
    </member>
    <member kind="function">
      <type>rocrand_status ROCRANDAPI</type>
      <name>rocrand_generate_uniform</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gabce19bf091ad9dd1d58ac389fb141af6</anchor>
      <arglist>(rocrand_generator generator, float *output_data, size_t n)</arglist>
    </member>
    <member kind="function">
      <type>rocrand_status ROCRANDAPI</type>
      <name>rocrand_generate_uniform_double</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gab8f6e7b40f0dc6373c0c79fd68c76b8c</anchor>
      <arglist>(rocrand_generator generator, double *output_data, size_t n)</arglist>
    </member>
    <member kind="function">
      <type>rocrand_status ROCRANDAPI</type>
      <name>rocrand_generate_uniform_half</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gaaa329ca0784ae76ebe2529ec36166267</anchor>
      <arglist>(rocrand_generator generator, half *output_data, size_t n)</arglist>
    </member>
    <member kind="function">
      <type>rocrand_status ROCRANDAPI</type>
      <name>rocrand_generate_normal</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>ga129d77afa101cdd647707251f554a1df</anchor>
      <arglist>(rocrand_generator generator, float *output_data, size_t n, float mean, float stddev)</arglist>
    </member>
    <member kind="function">
      <type>rocrand_status ROCRANDAPI</type>
      <name>rocrand_generate_normal_double</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>ga72af3e3f0207634bb5b240f07d81e406</anchor>
      <arglist>(rocrand_generator generator, double *output_data, size_t n, double mean, double stddev)</arglist>
    </member>
    <member kind="function">
      <type>rocrand_status ROCRANDAPI</type>
      <name>rocrand_generate_normal_half</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gaacff42dbc8601de4f86651ea9d9575cb</anchor>
      <arglist>(rocrand_generator generator, half *output_data, size_t n, half mean, half stddev)</arglist>
    </member>
    <member kind="function">
      <type>rocrand_status ROCRANDAPI</type>
      <name>rocrand_generate_log_normal</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>ga850572f66ae25ebb1301649af78d80d1</anchor>
      <arglist>(rocrand_generator generator, float *output_data, size_t n, float mean, float stddev)</arglist>
    </member>
    <member kind="function">
      <type>rocrand_status ROCRANDAPI</type>
      <name>rocrand_generate_log_normal_double</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gaf6c54363e2c9612f005a52367a57c983</anchor>
      <arglist>(rocrand_generator generator, double *output_data, size_t n, double mean, double stddev)</arglist>
    </member>
    <member kind="function">
      <type>rocrand_status ROCRANDAPI</type>
      <name>rocrand_generate_log_normal_half</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gaf8ef3c8b92c736b0633010a989ca9824</anchor>
      <arglist>(rocrand_generator generator, half *output_data, size_t n, half mean, half stddev)</arglist>
    </member>
    <member kind="function">
      <type>rocrand_status ROCRANDAPI</type>
      <name>rocrand_generate_poisson</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gace088e83d118e5cf683a82573bbe5bd7</anchor>
      <arglist>(rocrand_generator generator, unsigned int *output_data, size_t n, double lambda)</arglist>
    </member>
    <member kind="function">
      <type>rocrand_status ROCRANDAPI</type>
      <name>rocrand_initialize_generator</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>ga3fd2ae6565bd073ccd22a3476ba5be12</anchor>
      <arglist>(rocrand_generator generator)</arglist>
    </member>
    <member kind="function">
      <type>rocrand_status ROCRANDAPI</type>
      <name>rocrand_set_stream</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gaacb4167bfafd9e7511ffdaa19f59ac21</anchor>
      <arglist>(rocrand_generator generator, hipStream_t stream)</arglist>
    </member>
    <member kind="function">
      <type>rocrand_status ROCRANDAPI</type>
      <name>rocrand_set_seed</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gaf8dae4c6c5c83a97e1d67339e6eb202a</anchor>
      <arglist>(rocrand_generator generator, unsigned long long seed)</arglist>
    </member>
    <member kind="function">
      <type>rocrand_status ROCRANDAPI</type>
      <name>rocrand_set_seed_uint4</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gadf1111db21cb282ece74ce3d15c2562c</anchor>
      <arglist>(rocrand_generator generator, uint4 seed)</arglist>
    </member>
    <member kind="function">
      <type>rocrand_status ROCRANDAPI</type>
      <name>rocrand_set_offset</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>ga24815d9f073a0abc714479bef4ad34af</anchor>
      <arglist>(rocrand_generator generator, unsigned long long offset)</arglist>
    </member>
    <member kind="function">
      <type>rocrand_status ROCRANDAPI</type>
      <name>rocrand_set_ordering</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gacb1caa806c7bf42e024a4bb9057ce07e</anchor>
      <arglist>(rocrand_generator generator, rocrand_ordering order)</arglist>
    </member>
    <member kind="function">
      <type>rocrand_status ROCRANDAPI</type>
      <name>rocrand_set_quasi_random_generator_dimensions</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gad60207411c2a634aa851fc033c401180</anchor>
      <arglist>(rocrand_generator generator, unsigned int dimensions)</arglist>
    </member>
    <member kind="function">
      <type>rocrand_status ROCRANDAPI</type>
      <name>rocrand_get_version</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gada98d41f4545f7abbd0f1be2ba8ea176</anchor>
      <arglist>(int *version)</arglist>
    </member>
    <member kind="function">
      <type>rocrand_status ROCRANDAPI</type>
      <name>rocrand_create_poisson_distribution</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>ga02bb152d66cc67c799d217200de7ef90</anchor>
      <arglist>(double lambda, rocrand_discrete_distribution *discrete_distribution)</arglist>
    </member>
    <member kind="function">
      <type>rocrand_status ROCRANDAPI</type>
      <name>rocrand_create_discrete_distribution</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gac600927326dd87006ce2a7f24e609fa1</anchor>
      <arglist>(const double *probabilities, unsigned int size, unsigned int offset, rocrand_discrete_distribution *discrete_distribution)</arglist>
    </member>
    <member kind="function">
      <type>rocrand_status ROCRANDAPI</type>
      <name>rocrand_destroy_discrete_distribution</name>
      <anchorfile>group__rocrandhost.html</anchorfile>
      <anchor>gaf2e95cb1e61842a42801950f553d5750</anchor>
      <arglist>(rocrand_discrete_distribution discrete_distribution)</arglist>
    </member>
  </compound>
  <compound kind="group">
    <name>rocranddevice</name>
    <title>rocRAND device functions</title>
    <filename>group__rocranddevice.html</filename>
    <member kind="define">
      <type>#define</type>
      <name>ROCRAND_LFSR113_DEFAULT_SEED_X</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gad7b9a9998c875ff2fc450eb5fda8123c</anchor>
      <arglist></arglist>
    </member>
    <member kind="define">
      <type>#define</type>
      <name>ROCRAND_LFSR113_DEFAULT_SEED_Y</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga42b6caeb3146cf57d8095ba69b2faed6</anchor>
      <arglist></arglist>
    </member>
    <member kind="define">
      <type>#define</type>
      <name>ROCRAND_LFSR113_DEFAULT_SEED_Z</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gab01fbf3f2209f949b60ee0990b471007</anchor>
      <arglist></arglist>
    </member>
    <member kind="define">
      <type>#define</type>
      <name>ROCRAND_LFSR113_DEFAULT_SEED_W</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga4e427750c6a51bfd63a7f02cb6e62b1e</anchor>
      <arglist></arglist>
    </member>
    <member kind="define">
      <type>#define</type>
      <name>ROCRAND_MRG31K3P_DEFAULT_SEED</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga0ea93cf8d2d16d5fd7db45ada5ddac05</anchor>
      <arglist></arglist>
    </member>
    <member kind="define">
      <type>#define</type>
      <name>ROCRAND_MRG32K3A_DEFAULT_SEED</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga2b4b37e72c090e6d99373ff68d14c173</anchor>
      <arglist></arglist>
    </member>
    <member kind="define">
      <type>#define</type>
      <name>ROCRAND_PHILOX4x32_DEFAULT_SEED</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga545da3f91883ce1a2990a1ec8141d5fb</anchor>
      <arglist></arglist>
    </member>
    <member kind="define">
      <type>#define</type>
      <name>ROCRAND_XORWOW_DEFAULT_SEED</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gafb4aa9f4548403e34c00b271c7ef1a77</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand_discrete</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga25407118ffdf710eb72e3a46b6006822</anchor>
      <arglist>(rocrand_state_philox4x32_10 *state, const rocrand_discrete_distribution discrete_distribution)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS uint4</type>
      <name>rocrand_discrete4</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga1888f6bd2cf3a2db5caac832e86bd845</anchor>
      <arglist>(rocrand_state_philox4x32_10 *state, const rocrand_discrete_distribution discrete_distribution)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand_discrete</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gaf62e6e610897ed704ec0a39c8ce269eb</anchor>
      <arglist>(rocrand_state_mrg31k3p *state, const rocrand_discrete_distribution discrete_distribution)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand_discrete</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gac726e2c0125de06facfd283d13b2e5e7</anchor>
      <arglist>(rocrand_state_mrg32k3a *state, const rocrand_discrete_distribution discrete_distribution)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand_discrete</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gabbf189e591919160d3488f9629518756</anchor>
      <arglist>(rocrand_state_xorwow *state, const rocrand_discrete_distribution discrete_distribution)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand_discrete</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gab9a019b5df1371600e125637cb91ec8d</anchor>
      <arglist>(rocrand_state_mtgp32 *state, const rocrand_discrete_distribution discrete_distribution)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand_discrete</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gaf92d85ebfec9d0d872283cf65d4b2dc4</anchor>
      <arglist>(rocrand_state_sobol32 *state, const rocrand_discrete_distribution discrete_distribution)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand_discrete</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gabbd1476ebca1a0a2197951c309e41e8b</anchor>
      <arglist>(rocrand_state_scrambled_sobol32 *state, const rocrand_discrete_distribution discrete_distribution)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand_discrete</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga86e38424535187129d417a5c4cb42b1e</anchor>
      <arglist>(rocrand_state_sobol64 *state, const rocrand_discrete_distribution discrete_distribution)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand_discrete</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga33c9a934d9af2436638b66be60df0010</anchor>
      <arglist>(rocrand_state_scrambled_sobol64 *state, const rocrand_discrete_distribution discrete_distribution)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand_discrete</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gada87bfe6dbf4f7a8917b95de5c7ece51</anchor>
      <arglist>(rocrand_state_lfsr113 *state, const rocrand_discrete_distribution discrete_distribution)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand_discrete</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga4cf31cf20bad1bddf9e781981c9b3ac1</anchor>
      <arglist>(rocrand_state_threefry2x32_20 *state, const rocrand_discrete_distribution discrete_distribution)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand_discrete</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gaffdc0b3bf2c23b73c6e1d1fb83f828c3</anchor>
      <arglist>(rocrand_state_threefry2x64_20 *state, const rocrand_discrete_distribution discrete_distribution)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand_discrete</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gaf83c9b1f75768f802d77c77123295698</anchor>
      <arglist>(rocrand_state_threefry4x32_20 *state, const rocrand_discrete_distribution discrete_distribution)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand_discrete</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gafc32e2f624032ce45d0b6b48f5980119</anchor>
      <arglist>(rocrand_state_threefry4x64_20 *state, const rocrand_discrete_distribution discrete_distribution)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS void</type>
      <name>rocrand_init</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gaa07d4de8bf238cac4dc96940420f02da</anchor>
      <arglist>(const uint4 seed, const unsigned int subsequence, rocrand_state_lfsr113 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga4b007dc96fdb8696b0bce9370ad440fc</anchor>
      <arglist>(rocrand_state_lfsr113 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_log_normal</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gab40c9f94648b62150fbf492c15032d42</anchor>
      <arglist>(rocrand_state_philox4x32_10 *state, float mean, float stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float2</type>
      <name>rocrand_log_normal2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga2ef9a02cf6883cf3856e889cd0b52ef9</anchor>
      <arglist>(rocrand_state_philox4x32_10 *state, float mean, float stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float4</type>
      <name>rocrand_log_normal4</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga21c00971d1bfb7749313380093130a08</anchor>
      <arglist>(rocrand_state_philox4x32_10 *state, float mean, float stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_log_normal_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga1a910b0413a9905ca92b736f84bcd732</anchor>
      <arglist>(rocrand_state_philox4x32_10 *state, double mean, double stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double2</type>
      <name>rocrand_log_normal_double2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga087a44d7cfcae7455a2fdaedcfa142c7</anchor>
      <arglist>(rocrand_state_philox4x32_10 *state, double mean, double stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double4</type>
      <name>rocrand_log_normal_double4</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga66cf1a04e4536936c8a07f5ef9c6420c</anchor>
      <arglist>(rocrand_state_philox4x32_10 *state, double mean, double stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_log_normal</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gae64fa3adf3aba7e3444af9171f5037b0</anchor>
      <arglist>(rocrand_state_mrg31k3p *state, float mean, float stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float2</type>
      <name>rocrand_log_normal2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gaf06ddbceabb5db3c8ce16f609a903376</anchor>
      <arglist>(rocrand_state_mrg31k3p *state, float mean, float stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_log_normal_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gaa598125c98e50b15fcf731e592dafd22</anchor>
      <arglist>(rocrand_state_mrg31k3p *state, double mean, double stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double2</type>
      <name>rocrand_log_normal_double2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga06d90aa2af3a732c18e6884ae8013b52</anchor>
      <arglist>(rocrand_state_mrg31k3p *state, double mean, double stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_log_normal</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga7779e2dd4866447169799746565bd07b</anchor>
      <arglist>(rocrand_state_mrg32k3a *state, float mean, float stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float2</type>
      <name>rocrand_log_normal2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga014643199ae558aea87efb72f783395e</anchor>
      <arglist>(rocrand_state_mrg32k3a *state, float mean, float stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_log_normal_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gaf07a79a89bd09595fb125d329afe7f5e</anchor>
      <arglist>(rocrand_state_mrg32k3a *state, double mean, double stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double2</type>
      <name>rocrand_log_normal_double2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga9f8945108ef17a59dab4debd5464734a</anchor>
      <arglist>(rocrand_state_mrg32k3a *state, double mean, double stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_log_normal</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gabc5b50d158b68522640acec12b3c9e30</anchor>
      <arglist>(rocrand_state_xorwow *state, float mean, float stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float2</type>
      <name>rocrand_log_normal2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga9d6fd7bad70505ddf37ae8cd6a40b806</anchor>
      <arglist>(rocrand_state_xorwow *state, float mean, float stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_log_normal_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga637aa2214f824682fc68679409e86986</anchor>
      <arglist>(rocrand_state_xorwow *state, double mean, double stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double2</type>
      <name>rocrand_log_normal_double2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga20a910c8acf570d5521361db0508d74d</anchor>
      <arglist>(rocrand_state_xorwow *state, double mean, double stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_log_normal</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga8cc6e633722706bca86771a2d1da2561</anchor>
      <arglist>(rocrand_state_mtgp32 *state, float mean, float stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_log_normal_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga2c009f7b5a3b53b4f1854b704431e02d</anchor>
      <arglist>(rocrand_state_mtgp32 *state, double mean, double stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_log_normal</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga5e2433fd5d46674936b32d0cf5979304</anchor>
      <arglist>(rocrand_state_sobol32 *state, float mean, float stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_log_normal_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga733f02fefe47ec5645152445c18623ff</anchor>
      <arglist>(rocrand_state_sobol32 *state, double mean, double stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_log_normal</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga71908b4386153e89bef4c178b986f97b</anchor>
      <arglist>(rocrand_state_scrambled_sobol32 *state, float mean, float stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_log_normal_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga4c1a373d74bfb3082729766accc28b9c</anchor>
      <arglist>(rocrand_state_scrambled_sobol32 *state, double mean, double stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_log_normal</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga5830a66f67aad9dec49cab76d698cb8c</anchor>
      <arglist>(rocrand_state_sobol64 *state, float mean, float stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_log_normal_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga7447bd22c69368311691ada239481976</anchor>
      <arglist>(rocrand_state_sobol64 *state, double mean, double stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_log_normal</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gaca64954e6033020170fe3ffc045e1f3f</anchor>
      <arglist>(rocrand_state_scrambled_sobol64 *state, float mean, float stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_log_normal_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gadd17c2be23e21482d57b6b628d26d8ec</anchor>
      <arglist>(rocrand_state_scrambled_sobol64 *state, double mean, double stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_log_normal</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gadd6776bfb148c2ce4f86142f6ecadb95</anchor>
      <arglist>(rocrand_state_lfsr113 *state, float mean, float stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float2</type>
      <name>rocrand_log_normal2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga47979179d6ca59a5fac27aea074734da</anchor>
      <arglist>(rocrand_state_lfsr113 *state, float mean, float stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_log_normal_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga7c5f7d0083c18e353e5dc53a0e59984b</anchor>
      <arglist>(rocrand_state_lfsr113 *state, double mean, double stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double2</type>
      <name>rocrand_log_normal_double2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gac3a9f9c1ee9f055aa1105a13bae22f4c</anchor>
      <arglist>(rocrand_state_lfsr113 *state, double mean, double stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_log_normal</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga427b9888fe9cbb0ab78498c060e34ee8</anchor>
      <arglist>(rocrand_state_threefry2x32_20 *state, double mean, double stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float2</type>
      <name>rocrand_log_normal2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga9db87b3d3f1ed838ade2186162621377</anchor>
      <arglist>(rocrand_state_threefry2x32_20 *state, float mean, float stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_log_normal_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gae3004b71be98ffa320cbc86c17133c28</anchor>
      <arglist>(rocrand_state_threefry2x32_20 *state, double mean, double stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double2</type>
      <name>rocrand_log_normal_double2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga8698e3ab85f389c96266fff6a3edbad4</anchor>
      <arglist>(rocrand_state_threefry2x32_20 *state, double mean, double stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_log_normal</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gaaa18cd2acb3f251aa1d3bb125fd1108c</anchor>
      <arglist>(rocrand_state_threefry2x64_20 *state, double mean, double stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float2</type>
      <name>rocrand_log_normal2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga93c43aa40dde184d23e8e2f6d7302329</anchor>
      <arglist>(rocrand_state_threefry2x64_20 *state, float mean, float stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_log_normal_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gaa3ec8b31918fafb696a0e19eb3e062f1</anchor>
      <arglist>(rocrand_state_threefry2x64_20 *state, double mean, double stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double2</type>
      <name>rocrand_log_normal_double2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gae0156503227c1f3bfd3051eed42a9b64</anchor>
      <arglist>(rocrand_state_threefry2x64_20 *state, double mean, double stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_log_normal</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gaf69f77ad8b410d158f5ee0c62e10b140</anchor>
      <arglist>(rocrand_state_threefry4x32_20 *state, double mean, double stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float2</type>
      <name>rocrand_log_normal2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga79fa33620be165dca475ec760548f64e</anchor>
      <arglist>(rocrand_state_threefry4x32_20 *state, float mean, float stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_log_normal_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gac7f942df3a15adcd4aa5767e4b815b09</anchor>
      <arglist>(rocrand_state_threefry4x32_20 *state, double mean, double stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double2</type>
      <name>rocrand_log_normal_double2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga9b8ebd0200ffe09a28244f0dd37c965f</anchor>
      <arglist>(rocrand_state_threefry4x32_20 *state, double mean, double stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_log_normal</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga3c8d5d225e5218f2909382efed4dad18</anchor>
      <arglist>(rocrand_state_threefry4x64_20 *state, double mean, double stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float2</type>
      <name>rocrand_log_normal2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gab2123dd461e9b5541edb89d03e805a73</anchor>
      <arglist>(rocrand_state_threefry4x64_20 *state, float mean, float stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_log_normal_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga911f881671df3500e38b60e1b4855c1b</anchor>
      <arglist>(rocrand_state_threefry4x64_20 *state, double mean, double stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double2</type>
      <name>rocrand_log_normal_double2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gaebabfd255fd3eae72a4298aa71f17e6c</anchor>
      <arglist>(rocrand_state_threefry4x64_20 *state, double mean, double stddev)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS void</type>
      <name>rocrand_init</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga61fe0003d36da14a00515d3bf30cea94</anchor>
      <arglist>(const unsigned long long seed, const unsigned long long subsequence, const unsigned long long offset, rocrand_state_mrg31k3p *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga01e76819438325a4ebc729b33b4b5560</anchor>
      <arglist>(rocrand_state_mrg31k3p *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS void</type>
      <name>skipahead</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga7c64d54fdf493245b42b021f5d39c016</anchor>
      <arglist>(unsigned long long offset, rocrand_state_mrg31k3p *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS void</type>
      <name>skipahead_subsequence</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gac30bc4a20dabf8fa707a7883bae2e497</anchor>
      <arglist>(unsigned long long subsequence, rocrand_state_mrg31k3p *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS void</type>
      <name>skipahead_sequence</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga935ee353cb7f1370c8f183d54baf42c4</anchor>
      <arglist>(unsigned long long sequence, rocrand_state_mrg31k3p *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS void</type>
      <name>rocrand_init</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga02741290cb854aefcc31ca799d587cda</anchor>
      <arglist>(const unsigned long long seed, const unsigned long long subsequence, const unsigned long long offset, rocrand_state_mrg32k3a *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gac9330b64852ffa5790c0747894272fa9</anchor>
      <arglist>(rocrand_state_mrg32k3a *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS void</type>
      <name>skipahead</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gadb37aae5963a15cbffbddc0d561d7529</anchor>
      <arglist>(unsigned long long offset, rocrand_state_mrg32k3a *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS void</type>
      <name>skipahead_subsequence</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gab44395fa0083d68e568eb4bf825ef4f3</anchor>
      <arglist>(unsigned long long subsequence, rocrand_state_mrg32k3a *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS void</type>
      <name>skipahead_sequence</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga5da0ec1bfe3e2e5f8aeba56f0dbd267c</anchor>
      <arglist>(unsigned long long sequence, rocrand_state_mrg32k3a *state)</arglist>
    </member>
    <member kind="function">
      <type>__host__ rocrand_status</type>
      <name>rocrand_make_state_mtgp32</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga51c08c5f69bd79e41384260cd1a6a8ee</anchor>
      <arglist>(rocrand_state_mtgp32 *d_state, mtgp32_fast_params params[], int n, unsigned long long seed)</arglist>
    </member>
    <member kind="function">
      <type>__host__ rocrand_status</type>
      <name>rocrand_make_constant</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga814a21dd3196c0466029eaeb54f48b52</anchor>
      <arglist>(const mtgp32_fast_params params[], mtgp32_params *p)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gadd3276de91c3dbdae65cd36a1910b359</anchor>
      <arglist>(rocrand_state_mtgp32 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS void</type>
      <name>rocrand_mtgp32_block_copy</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga046293451b66fe9e729027193515d8e4</anchor>
      <arglist>(rocrand_state_mtgp32 *src, rocrand_state_mtgp32 *dest)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS void</type>
      <name>rocrand_mtgp32_set_params</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gaee4fab249fb1acf6895050423cf9a9f4</anchor>
      <arglist>(rocrand_state_mtgp32 *state, mtgp32_params *params)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_normal</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gaae5b2659b114b8e9cda1e2355ed86c90</anchor>
      <arglist>(rocrand_state_philox4x32_10 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float2</type>
      <name>rocrand_normal2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga090bee5a14830fe40d88c48f192df6b8</anchor>
      <arglist>(rocrand_state_philox4x32_10 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float4</type>
      <name>rocrand_normal4</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga826854709b738f6f2290069a450c0258</anchor>
      <arglist>(rocrand_state_philox4x32_10 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_normal_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga2430c3246f46154aa370ee7f02cc400a</anchor>
      <arglist>(rocrand_state_philox4x32_10 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double2</type>
      <name>rocrand_normal_double2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gae68753987a8ce135648ce8e971b8b877</anchor>
      <arglist>(rocrand_state_philox4x32_10 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double4</type>
      <name>rocrand_normal_double4</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga2650e2a840e8c89ba73434722c784e8f</anchor>
      <arglist>(rocrand_state_philox4x32_10 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_normal</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga5ee06f8eeca8cd52c131139049451b67</anchor>
      <arglist>(rocrand_state_mrg31k3p *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float2</type>
      <name>rocrand_normal2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga0a3f94aba52d54ec2e801b94c93ccdb7</anchor>
      <arglist>(rocrand_state_mrg31k3p *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_normal_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga0e84f2430a0561c12322bbec3679fed3</anchor>
      <arglist>(rocrand_state_mrg31k3p *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double2</type>
      <name>rocrand_normal_double2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga9052a9d9324b36cbe7c9355919a206f1</anchor>
      <arglist>(rocrand_state_mrg31k3p *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_normal</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga875b7abb296d442569af8c656c301e8b</anchor>
      <arglist>(rocrand_state_mrg32k3a *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float2</type>
      <name>rocrand_normal2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gae74ee82b180c8b1cd8602232b7764e1c</anchor>
      <arglist>(rocrand_state_mrg32k3a *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_normal_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga3fcb09ff5a906719dba4539cbf7be7ab</anchor>
      <arglist>(rocrand_state_mrg32k3a *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double2</type>
      <name>rocrand_normal_double2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gae66cb5ce86a31bf0f990437b5c86145a</anchor>
      <arglist>(rocrand_state_mrg32k3a *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_normal</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga9353d42cedb08ebaaa93aecea370275b</anchor>
      <arglist>(rocrand_state_xorwow *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float2</type>
      <name>rocrand_normal2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga9eabbc3294887aa41420703f6ab00611</anchor>
      <arglist>(rocrand_state_xorwow *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_normal_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga01cc87b7452294ba7deef944240bbacf</anchor>
      <arglist>(rocrand_state_xorwow *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double2</type>
      <name>rocrand_normal_double2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga2b7110d15e7142d8a7386cb3daa8591d</anchor>
      <arglist>(rocrand_state_xorwow *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_normal</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga2a56da6b4917db821f5d1e4983201b00</anchor>
      <arglist>(rocrand_state_mtgp32 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_normal_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga53202f24dcb723750477be469f1792fe</anchor>
      <arglist>(rocrand_state_mtgp32 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_normal</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga91a8c9a6c8face2dd8253c6b1e9a6b0d</anchor>
      <arglist>(rocrand_state_sobol32 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_normal_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga0688f77f03dccede0db4e267a66f4b37</anchor>
      <arglist>(rocrand_state_sobol32 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_normal</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gade9ec6b2db38d8fb3711f001e867ae21</anchor>
      <arglist>(rocrand_state_scrambled_sobol32 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_normal_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gadc920de89b09f30f64862ba594fb7b84</anchor>
      <arglist>(rocrand_state_scrambled_sobol32 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_normal</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga8f287814f2a3f0fa7b8a663c5b511672</anchor>
      <arglist>(rocrand_state_sobol64 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_normal_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gaa0fe4809ad9437925d40f8e9d3c536d8</anchor>
      <arglist>(rocrand_state_sobol64 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_normal</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga13e2d73c47957c1cb7e29eccaa8cbe88</anchor>
      <arglist>(rocrand_state_scrambled_sobol64 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_normal_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga88a54b775e25d335fc7708e18f25858e</anchor>
      <arglist>(rocrand_state_scrambled_sobol64 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_normal</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga8e963b80a6430b2f6c9d4fd873ab9961</anchor>
      <arglist>(rocrand_state_lfsr113 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float2</type>
      <name>rocrand_normal2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga93ab6cdf89410d8e9018c5a389fa38e4</anchor>
      <arglist>(rocrand_state_lfsr113 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_normal_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga73e90f8a4b413ac07536c3b7f6e78198</anchor>
      <arglist>(rocrand_state_lfsr113 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double2</type>
      <name>rocrand_normal_double2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga3e13ddf5c3ea592511f2a8d471f6aa8e</anchor>
      <arglist>(rocrand_state_lfsr113 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_normal</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga00e2cd4580aace9c3507d17be9981d0a</anchor>
      <arglist>(rocrand_state_threefry2x32_20 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float2</type>
      <name>rocrand_normal2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gae528d51c490e875164bf12bf76d0617f</anchor>
      <arglist>(rocrand_state_threefry2x32_20 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_normal_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga5a6c37448c1cfbfe1018ee71b4587235</anchor>
      <arglist>(rocrand_state_threefry2x32_20 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double2</type>
      <name>rocrand_normal_double2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga2e4fb5da3d1bfd4f2ad7c7c71ecf4298</anchor>
      <arglist>(rocrand_state_threefry2x32_20 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_normal</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga827bb7a715a1f71bab7202932bd8f2fe</anchor>
      <arglist>(rocrand_state_threefry2x64_20 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float2</type>
      <name>rocrand_normal2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga986d4d525ede3706852ae0d666a27bcd</anchor>
      <arglist>(rocrand_state_threefry2x64_20 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_normal_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga85632249cb5bd97d0ef923950371260c</anchor>
      <arglist>(rocrand_state_threefry2x64_20 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double2</type>
      <name>rocrand_normal_double2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gadb0a3c2149ddcdfd1193c800d6721c67</anchor>
      <arglist>(rocrand_state_threefry2x64_20 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_normal</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gaf8eb38f7f4b89e1589004e5f8c0f0e10</anchor>
      <arglist>(rocrand_state_threefry4x32_20 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float2</type>
      <name>rocrand_normal2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gaabb273178c2b4b23949784349569313b</anchor>
      <arglist>(rocrand_state_threefry4x32_20 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_normal_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga6b9a8a1feddee4954304b8f965a00fea</anchor>
      <arglist>(rocrand_state_threefry4x32_20 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double2</type>
      <name>rocrand_normal_double2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gabdf01ee586a6e6d13fdfebfd248a678d</anchor>
      <arglist>(rocrand_state_threefry4x32_20 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_normal</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga041766c2d6b2cf6d73698020d66d4644</anchor>
      <arglist>(rocrand_state_threefry4x64_20 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float2</type>
      <name>rocrand_normal2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gabfa64c202cd77975c3b5441d12c537c4</anchor>
      <arglist>(rocrand_state_threefry4x64_20 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_normal_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga5e8d56ddab0050f93821e8d4094e2e27</anchor>
      <arglist>(rocrand_state_threefry4x64_20 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double2</type>
      <name>rocrand_normal_double2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gab4770151a507b17da8f704ffb8f80c68</anchor>
      <arglist>(rocrand_state_threefry4x64_20 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS void</type>
      <name>rocrand_init</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gac2f06dd8828c39cc119b99cb1c30bebc</anchor>
      <arglist>(const unsigned long long seed, const unsigned long long subsequence, const unsigned long long offset, rocrand_state_philox4x32_10 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga0c58b214ee17ca1a3f066623b0c3ca4f</anchor>
      <arglist>(rocrand_state_philox4x32_10 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS uint4</type>
      <name>rocrand4</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gabe605f76b482184c0c04fc6ae5833fbf</anchor>
      <arglist>(rocrand_state_philox4x32_10 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS void</type>
      <name>skipahead</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gaceaaac5b17eeda19f410311957984495</anchor>
      <arglist>(unsigned long long offset, rocrand_state_philox4x32_10 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS void</type>
      <name>skipahead_subsequence</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga5f9274641be9285651808095c261cef4</anchor>
      <arglist>(unsigned long long subsequence, rocrand_state_philox4x32_10 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS void</type>
      <name>skipahead_sequence</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga95d878c7539452746622c2ead4521c2d</anchor>
      <arglist>(unsigned long long sequence, rocrand_state_philox4x32_10 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand_poisson</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gae193e88c203979f71871f8c97ca45655</anchor>
      <arglist>(rocrand_state_philox4x32_10 *state, double lambda)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS uint4</type>
      <name>rocrand_poisson4</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gada2c0b28bd5e80f7842d29e79caef035</anchor>
      <arglist>(rocrand_state_philox4x32_10 *state, double lambda)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand_poisson</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gad233395607ecc1fb0fc87e8a44343170</anchor>
      <arglist>(rocrand_state_mrg31k3p *state, double lambda)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand_poisson</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga621dc0990a24bea8b8ee3b261afc4712</anchor>
      <arglist>(rocrand_state_mrg32k3a *state, double lambda)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand_poisson</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga4274a70ca4a3c9ee1ff937288b5a493c</anchor>
      <arglist>(rocrand_state_xorwow *state, double lambda)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand_poisson</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga3c0672ddfabca191b3b22245712f85c6</anchor>
      <arglist>(rocrand_state_mtgp32 *state, double lambda)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand_poisson</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga700eca5b0ea504b97b86af0ee74878ac</anchor>
      <arglist>(rocrand_state_sobol32 *state, double lambda)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand_poisson</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga92f8ee127fcc5ef29ac7ccf19fdd3aa3</anchor>
      <arglist>(rocrand_state_scrambled_sobol32 *state, double lambda)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned long long int</type>
      <name>rocrand_poisson</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gac7cbcc5d13a5704923ea0633468082ad</anchor>
      <arglist>(rocrand_state_sobol64 *state, double lambda)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned long long int</type>
      <name>rocrand_poisson</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga69b5b7bd1614e2776a80ffe31cf5c37e</anchor>
      <arglist>(rocrand_state_scrambled_sobol64 *state, double lambda)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand_poisson</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga81daf2faa36a465430d6e76f86146aa1</anchor>
      <arglist>(rocrand_state_lfsr113 *state, double lambda)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand_poisson</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gaf13b00ef7dc19614408b445f86f90a59</anchor>
      <arglist>(rocrand_state_threefry2x32_20 *state, double lambda)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand_poisson</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga1e87de0dd83190c479c51071497f1057</anchor>
      <arglist>(rocrand_state_threefry2x64_20 *state, double lambda)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand_poisson</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gadff0ca3a60fffbadd6bc6793908b9686</anchor>
      <arglist>(rocrand_state_threefry4x32_20 *state, double lambda)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand_poisson</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga8dda599d6eea7afa58ca83c8b85d6dae</anchor>
      <arglist>(rocrand_state_threefry4x64_20 *state, double lambda)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS void</type>
      <name>rocrand_init</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga9604277f246a78254bc4f9fd81e19cda</anchor>
      <arglist>(const unsigned int *vectors, const unsigned int scramble_constant, const unsigned int offset, rocrand_state_scrambled_sobol32 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga56ab665a27f9444404312b2bbd15cb65</anchor>
      <arglist>(rocrand_state_scrambled_sobol32 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS void</type>
      <name>skipahead</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga2513ad5e30cfadfd2557bafc00767727</anchor>
      <arglist>(unsigned long long offset, rocrand_state_scrambled_sobol32 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS void</type>
      <name>rocrand_init</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gad5eb6d1edd1ebb5dbe76b3672156ba8d</anchor>
      <arglist>(const unsigned long long int *vectors, const unsigned long long int scramble_constant, const unsigned int offset, rocrand_state_scrambled_sobol64 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned long long int</type>
      <name>rocrand</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga8d27201d6c0a075985107b6b6b02b8d5</anchor>
      <arglist>(rocrand_state_scrambled_sobol64 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS void</type>
      <name>skipahead</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga251d4e1ad9d93f13935ff12f79d140ab</anchor>
      <arglist>(unsigned long long offset, rocrand_state_scrambled_sobol64 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS void</type>
      <name>rocrand_init</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga5ca43f601b6fbc218312fe1ba102089b</anchor>
      <arglist>(const unsigned int *vectors, const unsigned int offset, rocrand_state_sobol32 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga57b412c7864733385133f87e00dd731e</anchor>
      <arglist>(rocrand_state_sobol32 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS void</type>
      <name>skipahead</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga00519fcb90b04712e61b6198bb91b59f</anchor>
      <arglist>(unsigned long long offset, rocrand_state_sobol32 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS void</type>
      <name>rocrand_init</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga39af4f8242e72f4d5cc197b5600e7369</anchor>
      <arglist>(const unsigned long long int *vectors, const unsigned int offset, rocrand_state_sobol64 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned long long int</type>
      <name>rocrand</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga27dc0e6d6992eec49039506c471a57fb</anchor>
      <arglist>(rocrand_state_sobol64 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS void</type>
      <name>skipahead</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gac638c1e10be3725b53620858eda8860a</anchor>
      <arglist>(unsigned long long int offset, rocrand_state_sobol64 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_uniform</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga9d8fb3aad22a13c0cb3944b59eb6bb9a</anchor>
      <arglist>(rocrand_state_philox4x32_10 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float2</type>
      <name>rocrand_uniform2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga13c610436e8d20e8411b0c592541c13b</anchor>
      <arglist>(rocrand_state_philox4x32_10 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float4</type>
      <name>rocrand_uniform4</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gae3f5d05afa1ddf66fb2ae88b861a0c34</anchor>
      <arglist>(rocrand_state_philox4x32_10 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_uniform_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga41e64103b2362674735bd7bbc2534f86</anchor>
      <arglist>(rocrand_state_philox4x32_10 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double2</type>
      <name>rocrand_uniform_double2</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gae700bb333a2158660ee4c0e0acf50edb</anchor>
      <arglist>(rocrand_state_philox4x32_10 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double4</type>
      <name>rocrand_uniform_double4</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga369fda238036adbe288bf2c1b8e39214</anchor>
      <arglist>(rocrand_state_philox4x32_10 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_uniform</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga7e9b7b845bd7abe50b44c0eee83e4f67</anchor>
      <arglist>(rocrand_state_mrg31k3p *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_uniform_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga7139e76efe0d1c7682d2fc94ad6c927e</anchor>
      <arglist>(rocrand_state_mrg31k3p *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_uniform</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gae96bbe61b8e21f2dd40c5ba0cd5e0ce9</anchor>
      <arglist>(rocrand_state_mrg32k3a *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_uniform_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gae8b555fdd22fbb6b6d2fc850983a4b73</anchor>
      <arglist>(rocrand_state_mrg32k3a *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_uniform</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga9ce802447438e90e709ed20e95fb7f95</anchor>
      <arglist>(rocrand_state_xorwow *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_uniform_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga3c236a5f015bad53b42713cc87787de8</anchor>
      <arglist>(rocrand_state_xorwow *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_uniform</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga3f5ed873cece7fdf0efa8bca4979aef3</anchor>
      <arglist>(rocrand_state_mtgp32 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_uniform_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga488aa98c8ccc728483989a2556f17c98</anchor>
      <arglist>(rocrand_state_mtgp32 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_uniform</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gac6ac50d50c2b29836950af1a94611df8</anchor>
      <arglist>(rocrand_state_sobol32 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_uniform_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gabb683cf251fe2f9e0fc696d82eca8a32</anchor>
      <arglist>(rocrand_state_sobol32 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_uniform</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga8fa0721baf710856564b49d9079509f9</anchor>
      <arglist>(rocrand_state_scrambled_sobol32 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_uniform_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gaffc772664d75d936cab69c5ac3fed9b9</anchor>
      <arglist>(rocrand_state_scrambled_sobol32 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_uniform</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga0998a4e2dd20ae1440378645e1c560ac</anchor>
      <arglist>(rocrand_state_sobol64 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_uniform_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga161ed44e86dc603a396638c687ce3b74</anchor>
      <arglist>(rocrand_state_sobol64 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_uniform</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga33f8dfded4f6914ef47fe74f753a4853</anchor>
      <arglist>(rocrand_state_scrambled_sobol64 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_uniform_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gace65e9d1180e6a6eb0379846b8066a28</anchor>
      <arglist>(rocrand_state_scrambled_sobol64 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_uniform</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga832d9c3f451e0df63943f242f64f4b8e</anchor>
      <arglist>(rocrand_state_lfsr113 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_uniform_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gabdaf9266cef05e62fa24eea8adff83f8</anchor>
      <arglist>(rocrand_state_lfsr113 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_uniform</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gab0066f1f2a85e18e8ce61eb03cb3be89</anchor>
      <arglist>(rocrand_state_threefry2x32_20 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_uniform_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga4f5ba240fd3d74005519bd31e6493b83</anchor>
      <arglist>(rocrand_state_threefry2x32_20 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_uniform</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga35241dd6e1396f513fc6c48019c0551d</anchor>
      <arglist>(rocrand_state_threefry2x64_20 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_uniform_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gaee59cc2cd777f1fd2f9b1cde958e2c42</anchor>
      <arglist>(rocrand_state_threefry2x64_20 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_uniform</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga90abbfedae6ab3521b45cc2f0b744492</anchor>
      <arglist>(rocrand_state_threefry4x32_20 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_uniform_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gabedfbf6a2db5465101bddb5d898dc73a</anchor>
      <arglist>(rocrand_state_threefry4x32_20 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS float</type>
      <name>rocrand_uniform</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gaa37999183ea5123a9b971cd228a35548</anchor>
      <arglist>(rocrand_state_threefry4x64_20 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS double</type>
      <name>rocrand_uniform_double</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga80f887a5b80326510af27519252109ac</anchor>
      <arglist>(rocrand_state_threefry4x64_20 *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS void</type>
      <name>rocrand_init</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gaa50d66f74cecb08db6f8c033939f1f8b</anchor>
      <arglist>(const unsigned long long seed, const unsigned long long subsequence, const unsigned long long offset, rocrand_state_xorwow *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS unsigned int</type>
      <name>rocrand</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga9f52e362aba35865ac497844535f50ba</anchor>
      <arglist>(rocrand_state_xorwow *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS void</type>
      <name>skipahead</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gaac7954b2d85ea7dc61d77a6a74e87e24</anchor>
      <arglist>(unsigned long long offset, rocrand_state_xorwow *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS void</type>
      <name>skipahead_subsequence</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>ga6455ec3c2a5ad67c70fbf679e0251a17</anchor>
      <arglist>(unsigned long long subsequence, rocrand_state_xorwow *state)</arglist>
    </member>
    <member kind="function">
      <type>FQUALIFIERS void</type>
      <name>skipahead_sequence</name>
      <anchorfile>group__rocranddevice.html</anchorfile>
      <anchor>gafcf0eec5952ab4e8edbdfbb3f7f0d86e</anchor>
      <arglist>(unsigned long long sequence, rocrand_state_xorwow *state)</arglist>
    </member>
    <page>group__rocranddevice</page>
  </compound>
  <compound kind="group">
    <name>rocrandhostcpp</name>
    <title>rocRAND host API C++ Wrapper</title>
    <filename>group__rocrandhostcpp.html</filename>
    <class kind="class">rocrand_cpp::error</class>
    <class kind="class">rocrand_cpp::uniform_int_distribution</class>
    <class kind="class">rocrand_cpp::uniform_real_distribution</class>
    <class kind="class">rocrand_cpp::normal_distribution</class>
    <class kind="class">rocrand_cpp::lognormal_distribution</class>
    <class kind="class">rocrand_cpp::poisson_distribution</class>
    <class kind="class">rocrand_cpp::philox4x32_10_engine</class>
    <class kind="class">rocrand_cpp::xorwow_engine</class>
    <class kind="class">rocrand_cpp::mrg31k3p_engine</class>
    <class kind="class">rocrand_cpp::mrg32k3a_engine</class>
    <class kind="class">rocrand_cpp::mtgp32_engine</class>
    <class kind="class">rocrand_cpp::lfsr113_engine</class>
    <class kind="class">rocrand_cpp::mt19937_engine</class>
    <class kind="class">rocrand_cpp::sobol32_engine</class>
    <class kind="class">rocrand_cpp::scrambled_sobol32_engine</class>
    <class kind="class">rocrand_cpp::sobol64_engine</class>
    <class kind="class">rocrand_cpp::scrambled_sobol64_engine</class>
    <class kind="class">rocrand_cpp::threefry2x32_20_engine</class>
    <class kind="class">rocrand_cpp::threefry2x64_20_engine</class>
    <class kind="class">rocrand_cpp::threefry4x32_20_engine</class>
    <class kind="class">rocrand_cpp::threefry4x64_20_engine</class>
    <member kind="typedef">
      <type>philox4x32_10_engine</type>
      <name>philox4x32_10</name>
      <anchorfile>group__rocrandhostcpp.html</anchorfile>
      <anchor>gaed266730800c29167fef57ecffc766cf</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>xorwow_engine</type>
      <name>xorwow</name>
      <anchorfile>group__rocrandhostcpp.html</anchorfile>
      <anchor>gabc91fa4ea7737363c8db15b2e22a4a3f</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>mrg31k3p_engine</type>
      <name>mrg31k3a</name>
      <anchorfile>group__rocrandhostcpp.html</anchorfile>
      <anchor>ga55899bdd90e93ae58bbe7309f6757f09</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>mrg32k3a_engine</type>
      <name>mrg32k3a</name>
      <anchorfile>group__rocrandhostcpp.html</anchorfile>
      <anchor>ga41e4586d94c436c2072fc1104135befd</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>mtgp32_engine</type>
      <name>mtgp32</name>
      <anchorfile>group__rocrandhostcpp.html</anchorfile>
      <anchor>gaf97fad61d2ae3c7033584faa63ba957d</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>lfsr113_engine</type>
      <name>lfsr113</name>
      <anchorfile>group__rocrandhostcpp.html</anchorfile>
      <anchor>gab2554c6af0f70f157eabc863dd7beb1e</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>mt19937_engine</type>
      <name>mt19937</name>
      <anchorfile>group__rocrandhostcpp.html</anchorfile>
      <anchor>ga9e9497e330fa717e2b502d0873de65be</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>threefry2x32_20_engine</type>
      <name>threefry2x32</name>
      <anchorfile>group__rocrandhostcpp.html</anchorfile>
      <anchor>ga16f2926fa3f9ae67dda5a55779d8de51</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>threefry2x64_20_engine</type>
      <name>threefry2x64</name>
      <anchorfile>group__rocrandhostcpp.html</anchorfile>
      <anchor>gae4497a2fa6769dc6736d00f8e79a1ec1</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>threefry4x32_20_engine</type>
      <name>threefry4x32</name>
      <anchorfile>group__rocrandhostcpp.html</anchorfile>
      <anchor>gad03bed41243ccf8281eb20bf0c5492c8</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>threefry4x64_20_engine</type>
      <name>threefry4x64</name>
      <anchorfile>group__rocrandhostcpp.html</anchorfile>
      <anchor>gab7652524bd1228a16e0dede27f9db24f</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>sobol32_engine</type>
      <name>sobol32</name>
      <anchorfile>group__rocrandhostcpp.html</anchorfile>
      <anchor>gaac6c97f286f1eafbb181e7c0e80ed682</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>scrambled_sobol32_engine</type>
      <name>scrambled_sobol32</name>
      <anchorfile>group__rocrandhostcpp.html</anchorfile>
      <anchor>ga5d2370cc545a4bf0f0f8ef3a20d21de7</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>sobol64_engine</type>
      <name>sobol64</name>
      <anchorfile>group__rocrandhostcpp.html</anchorfile>
      <anchor>ga75ce3a14366d873ea9c65ffc78fd6093</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>scrambled_sobol64_engine</type>
      <name>scrambled_sobol64</name>
      <anchorfile>group__rocrandhostcpp.html</anchorfile>
      <anchor>ga4bf3ebb685267a05859100aa9ea7dca1</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>xorwow</type>
      <name>default_random_engine</name>
      <anchorfile>group__rocrandhostcpp.html</anchorfile>
      <anchor>gaeb5d7c54dab07c51c6f241585543c1e3</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>std::random_device</type>
      <name>random_device</name>
      <anchorfile>group__rocrandhostcpp.html</anchorfile>
      <anchor>gafa11687825531fab552a5d8e075417b7</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type>int</type>
      <name>version</name>
      <anchorfile>group__rocrandhostcpp.html</anchorfile>
      <anchor>ga3ac75ea7bbd4ade1a4fdabdf2a629201</anchor>
      <arglist>()</arglist>
    </member>
  </compound>
  <compound kind="group">
    <name>hiprandhost</name>
    <title>hipRAND host API</title>
    <filename>group__hiprandhost.html</filename>
  </compound>
  <compound kind="group">
    <name>hipranddevice</name>
    <title>hipRAND device functions</title>
    <filename>group__hipranddevice.html</filename>
    <page>group__hipranddevice</page>
  </compound>
  <compound kind="group">
    <name>hiprandhostcpp</name>
    <title>hipRAND host API C++ Wrapper</title>
    <filename>group__hiprandhostcpp.html</filename>
  </compound>
  <compound kind="page">
    <name>rocranddevice_page</name>
    <title>rocRAND RNG&apos;s state types</title>
    <filename>group__rocranddevice</filename>
  </compound>
  <compound kind="page">
    <name>hipranddevice_page</name>
    <title>hipRAND RNG&apos;s state types</title>
    <filename>group__hipranddevice</filename>
  </compound>
  <compound kind="page">
    <name>index</name>
    <title>rocRAND Documentation</title>
    <filename>index</filename>
    <docanchor file="index.html" title="Overview">overview</docanchor>
  </compound>
</tagfile>
