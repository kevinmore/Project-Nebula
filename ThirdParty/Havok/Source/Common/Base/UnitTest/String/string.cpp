/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Container/String/hkUtf8.h>

static const char helloworld[] = "Hello world!\n";
static const char HELLOWORLD[] = "HELLO WORLD!\n";
static const char hello[] = "Hello";
static const char world[] = "world!\n";

static void hkarray_stringbuf_not_allowed()
{
	// uncommenting should produce a compile error
	//hkArray<hkStringBuf> sb0;
}

static void string_memset16_test()
{
	HK_ALIGN16(hkUint32 dst[1024 / 4]);
	HK_ALIGN16(hkUint32 src[   4]);

	src[0] = 0;
	src[1] = 1;
	src[2] = 2;
	src[3] = 3;

	// medium, gets unrolled on PlayStation(R)3
	hkString::memSet16<128> (dst, src);

	for (int i=0; i < (128 / 4); i+=4)
	{
		HK_TEST( dst[i+0] == 0 );
		HK_TEST( dst[i+1] == 1 );
		HK_TEST( dst[i+2] == 2 );
		HK_TEST( dst[i+3] == 3 );
	}

	src[0] = 10;
	src[1] = 11;
	src[2] = 12;
	src[3] = 13;

	// large, always falls back to loop
	hkString::memSet16<1024> (dst, src);

	for (int i=0; i < (1024 / 4); i+=4)
	{
		HK_TEST( dst[i+0] == 10 );
		HK_TEST( dst[i+1] == 11 );
		HK_TEST( dst[i+2] == 12 );
		HK_TEST( dst[i+3] == 13 );
	}
}

static void testSplitEqual(const char* src, const char*const* expected)
{
	hkStringBuf sb(src);
	hkArray<const char*>::Temp bits;
	sb.split('/', bits);
	int i;
	for( i = 0; i < bits.getSize(); ++i )
	{
		HK_TEST( expected[i] != HK_NULL );
		HK_TEST( hkString::strCmp(bits[i], expected[i]) == 0 );
	}
	HK_TEST(expected[i] == HK_NULL);
}

//#include <Common/Base/Fwd/hkcstdlib.h>
static void string_int24w()
{
#if 0 // currently fails
	hkVector4 v0;
	for( int i = 0; i < 1<<24; ++i )
	{
		v0.setInt24W(i);
		hkStringBuf sb;
		sb.printf("%g", v0(3));
		
		hkVector4 v1; v1(3) = (hkReal)strtod(sb.cString(), HK_NULL);
		HK_TEST2( v0(3) == v1(3), "Iteration" << i );
	}
#endif
}

static void testPathAppend( const char* expected, const char* orig, const char* a, const char* b)
{
	hkStringBuf sb = orig;
	sb.pathAppend(a,b);
	HK_TEST2( sb == expected, "PathAppend sb='"<<sb<<"' expected='"<<expected<<"'");
}

static void testPathNormalize( const char* input, const char* expected)
{
	hkStringBuf sb = input;
	sb.pathNormalize();
	HK_TEST2( sb == expected, "PathNormalize input='"<<input<<"' sb='"<<sb<<"' expected='"<<expected<<"'");
}

static void testStringsEqual(const wchar_t* wideInput, const char* utf8Input)
{
	// test that wild converts to utf8
	hkStringPtr utf8Local = hkUtf8::Utf8FromWide(wideInput).cString();
	HK_TEST( hkString::strCmp(utf8Input, utf8Local) == 0 );

	// and vice versa
	hkUtf8::WideFromUtf8 wideHelper(utf8Input);
	HK_TEST( hkString::memCmp(wideInput, wideHelper.cString(), wideHelper.getArray().getSize() * sizeof(wchar_t) ) == 0 );
}

static void testUtf8Decode(const wchar_t* wideInput, const char* utf8Input)
{
	// test that utf8 decodes to wide
	hkUtf8::WideFromUtf8 wideHelper(utf8Input);
	HK_TEST( hkString::memCmp(wideInput, wideHelper.cString(), wideHelper.getArray().getSize() * sizeof(wchar_t) ) == 0 );
}

static void string_encoding_equal()
{
	//Visual Studio doesn't do string unicode constants so we use escapes
	testStringsEqual(L"\x3072\x307f\x3087 \x3074\x30b8", "\xe3\x81\xb2\xe3\x81\xbf\xe3\x82\x87 \xe3\x81\xb4\xe3\x82\xb8");

	// test bad encodings

	testUtf8Decode(L"bad \xfffd yeah?", "bad \xff yeah?"); // non utf8 lead char

	// Check invalid encodings: L(ength of sequence) A(ctual length)
	testUtf8Decode(L"bad \xfffd *", "bad \xca *"); // L2 A1
	testUtf8Decode(L"bad \xfffd *", "bad \xea\x8a *"); // L3 A2
	testUtf8Decode(L"bad \xfffd *", "bad \xf7\x8a *"); // L4 A2
	testUtf8Decode(L"bad \xfffd *", "bad \xfd\x8f\x8f *"); // L5 A3
	testUtf8Decode(L"bad \xfffd\x3074 *", "bad \xfd\x8f\x8f\xe3\x81\xb4 *"); // L5 A3 then valid utf8 (starts with e3)

	{
		wchar_t wbuf[100] = {};
		int n;
		n = hkUtf8::wideFromUtf8(HK_NULL, 0, "hello");
		HK_TEST( n == 6 );
		n = hkUtf8::wideFromUtf8(wbuf, 100, "hello");
		HK_TEST( n == 6 );
		HK_TEST( hkString::memCmp(wbuf, L"hello", 6*sizeof(wchar_t)) == 0);

		// only room for 1st char
		n = hkUtf8::wideFromUtf8(wbuf, 2, "<\xe3\x81\xb4>");
		HK_TEST( n == 4 );
		HK_TEST( wbuf[1] == 0 );
	}

	{
		char cbuf[100] = {};
		const wchar_t wsun[] = L"<\x65e5>";
		const char csun[] = "<\xe6\x97\xa5>";
		int n8 = hkUtf8::utf8FromWide(HK_NULL, 0, wsun);
		HK_TEST( n8 == sizeof(csun) ); //0x65e5 -> 0xE6 0x97 0xA5
		n8 = hkUtf8::utf8FromWide(cbuf, 100, wsun);
		HK_TEST( hkString::memCmp(cbuf, csun, sizeof(csun)) == 0 );
		// room for first but not enough room for 2nd char
		n8 = hkUtf8::utf8FromWide(cbuf, 3, wsun);
		HK_TEST( n8 == sizeof(csun) );
		HK_TEST( cbuf[0]=='<' );
		HK_TEST( cbuf[1]==0 );
	}

}



int string_main()
{
	{
		hkStringBuf s;
		HK_TEST(s.cString()[0] == 0);
		HK_TEST(s.getLength() == 0);
	}
	{
		hkStringBuf s(helloworld);
		HK_TEST(s[0]=='H');
		HK_TEST(s.getLength() == sizeof(helloworld) - 1);

		HK_TEST(s.startsWith(hello));
		hkStringBuf hello2(hello);
		HK_TEST(s.startsWith(hello2));

		HK_TEST(s.endsWith(world));
		hkStringBuf world2(world);
		HK_TEST(s.endsWith(world2));

		HK_TEST(s.indexOf('l') == 2 );
		HK_TEST(s.indexOf('!') == 11);

		{
			hkStringBuf scpy( s );
			scpy.upperCase();
			HK_TEST( scpy == HELLOWORLD );
		}
		hkStringBuf t( HELLOWORLD );
		{
			hkStringBuf tcpy( t );
			tcpy.lowerCase();
			HK_TEST( tcpy != s );
		}

		HK_TEST( s.compareToIgnoreCase(t)==0 );

		{
			hkStringBuf scpy( s );
			scpy.slice(3,5);
			HK_TEST( scpy == "lo wo");
		}
		
		HK_TEST( hkStringBuf("apples").compareTo("oranges") < 0 );
		HK_TEST( hkStringBuf("apples").compareTo("apples") == 0 );
		HK_TEST( hkStringBuf("bananas").compareTo("apples") > 0 );

		{
			hkStringBuf tfc("the", " fat", " cat");
			HK_TEST( tfc == "the fat cat" );
		}

		hkStringBuf u("one");
		u += "two";
		u += "three";
		HK_TEST( u == "onetwothree" );
	}
	{
		hkStringBuf s1(helloworld); s1.replace('o', 'X', hkStringBuf::REPLACE_ONE);
		hkStringBuf s2(helloworld); s2.replace('o', 'X', hkStringBuf::REPLACE_ALL);
		hkStringBuf s3(helloworld); s3.replace('o', 'X');
		HK_TEST( s1 == "HellX world!\n" );
		HK_TEST( s2 == "HellX wXrld!\n" );
		HK_TEST( s3 == "HellX wXrld!\n" );

		s1 = "this is the longest string you can think of or at least the longest I want to write.not short";
		s1 = "very short";
		s1.replace("very", "this is the longest string you can think of or at least the longest I want to write.not");
		HK_TEST( s1 == "this is the longest string you can think of or at least the longest I want to write.not short" );
	}
	{
		hkStringBuf s1("Lo! hello, hello!");
		hkStringBuf s2( s1 );
		hkStringBuf s3( s1 );

		s1.replace("lo", "[0123]", hkStringBuf::REPLACE_ONE);
		s2.replace("lo", "[0123]", hkStringBuf::REPLACE_ALL);
		s3.replace("lo", "[0123]"); 
		HK_TEST( s1 == "Lo! hel[0123], hello!" );
		HK_TEST( s2 == "Lo! hel[0123], hel[0123]!" );
		HK_TEST( s3 == "Lo! hel[0123], hel[0123]!" );
	}
	{
		hkStringBuf s("Lo! hello, hello!");
		hkStringBuf s1( s ); 
		hkStringBuf s2 = s1; 
		s1.slice(0,3); 
		s2.slice(4,5); 
		HK_TEST( s1 == "Lo!");
		HK_TEST( s2 == "hello");
	}

	{
		// standard
		HK_TEST( hkString::atoi("123") == 123);
		HK_TEST( hkString::atoi("-123") == -123);

		// Unary + is allowed
		HK_TEST( hkString::atoi("+123") == 123);

		// Whitespace
		HK_TEST( hkString::atoi(" 123") == 123);
		HK_TEST( hkString::atoi("  123") == 123);
		HK_TEST( hkString::atoi("\t123") == 123);
		HK_TEST( hkString::atoi("\t\t123") == 123);

		// Garbage allowed at end
		HK_TEST( hkString::atoi("123BAD") == 123);

		// Other bases
		HK_TEST( hkString::atoi("0x400") == 1024);

		// Uber test
		HK_TEST( hkString::atoi("  \t  \t +123BAD") == 123);

	}

	{
		// standard
		HK_TEST( 0			== hkString::atof( "0" ) );
		HK_TEST( 5			== hkString::atof( "5" ) );
		HK_TEST( -5			== hkString::atof( "-5" ) );
		HK_TEST( 0			== hkString::atof( "0.0" ) );
		HK_TEST( 5.05f - hkString::atof( "5.05" ) < 1e-6f);
		HK_TEST( 5.0f		== hkString::atof( "5." ) );
		HK_TEST( -0.5f		== hkString::atof( "-.5" ) );
		HK_TEST( -0.5f		== hkString::atof( "-0.5" ) );
		HK_TEST( 5.5f		== hkString::atof( "5.5" ) );
		HK_TEST( -5.5f		== hkString::atof( "-5.5" ) );
		HK_TEST( 500000.0f	== hkString::atof( "5e5" ) );
		HK_TEST( 500000.0f	== hkString::atof( "5E5" ) );
		HK_TEST( 0.0f		== hkString::atof( "0.0e5" ) ); 
		HK_TEST( 500000.0f	== hkString::atof( "5.e5" ) );
		HK_TEST( 50000.0f	== hkString::atof( ".5e5" ) );
		HK_TEST( -550000.0f == hkString::atof( "-5.5e5" ) );
//			HK_TEST( 0.001f > hkMath::fabs( 5e-5f - hkString::atof( "5e-5" ) ) );
//			HK_TEST( 0.001f > hkMath::fabs( 5e-5f - hkString::atof( "5E-5" ) ) );
//			HK_TEST( 0.001f > hkMath::fabs( -5e-5f - hkString::atof( "-5e-5" ) ) );
//			HK_TEST( 0.001f > hkMath::fabs( -5e-5f - hkString::atof( "-5E-5" ) ) );
//			HK_TEST( 0.001f > hkMath::fabs( 5e-5f - hkString::atof( "5.0e-5" ) ) );
//			HK_TEST( 0.001f > hkMath::fabs( 5e-6f - hkString::atof( "0.5e-5" ) ) );
//			HK_TEST( 0.001f > hkMath::fabs( -5.5e-5f - hkString::atof( "-5.5e-5" ) ) );
		
		// error really but does return result
		HK_TEST( 0			== hkString::atof( "" ) );
//			These tests seem to be invalid.
//			HK_TEST( 0.001f > hkMath::fabs( -5.5e-5f - hkString::atof( "-5.-5e-5" ) ) ); 
//			HK_TEST( -550000.0f == hkString::atof( "-5.-5e5" ) ); 
//			HK_TEST( -5.5f		== hkString::atof( "-5.-5" ) );
	}

	{
		char string1[] = "a";
		char string2[] = "AB";
		HK_TEST( hkString::strCasecmp(string1, string2) == -1 );
		HK_TEST( hkString::strCasecmp(string2, string1) == 1 );

		char string3[] = "ab";
		char string4[] = "AB";
		HK_TEST( hkString::strCasecmp(string3, string4) == 0 );
		HK_TEST( hkString::strCasecmp(string4, string3) == 0 );
	}
	{
		char string1[] = "ab";
		char string2[] = "ABC";
		int n = 1;
		HK_TEST( hkString::strNcasecmp(string1, string2, n) == 0 );
		HK_TEST( hkString::strNcasecmp(string2, string1, n) == 0 );

		n = 2;
		HK_TEST( hkString::strNcasecmp(string1, string2, n) == 0 );
		HK_TEST( hkString::strNcasecmp(string2, string1, n) == 0 );		

		n = 3;
		HK_TEST( hkString::strNcasecmp(string1, string2, n) == -1 );
		HK_TEST( hkString::strNcasecmp(string2, string1, n) == 1 );		

		n = 4;
		HK_TEST( hkString::strNcasecmp(string1, string2, n) == -1 );
		HK_TEST( hkString::strNcasecmp(string2, string1, n) == 1 );
	}
	{
		HK_TEST( hkString::indexOf("foo", 'x') == -1 );
		HK_TEST( hkString::indexOf("foo", 'o', 100, 1000) == -1 );
		HK_TEST( hkString::indexOf("foo", 'o', 1) == 1 );
		HK_TEST( hkString::indexOf("foo", 'o', 2) == 2 );
		HK_TEST( hkString::indexOf("foo", 'o', 0,1) == -1 );
		HK_TEST( hkString::indexOf("foo", 'o', 1,1) == -1 );
	}
	{
		hkStringBuf roman;
		roman.printf( "Write %s if you want %i.", "IV", 4 );
		HK_TEST( roman == "Write IV if you want 4." );
	}

	{
		hkStringPtr greeting = "hello";
		hkStringPtr goodbye = "goodbye";
		hkStringPtr null;
		
		HK_TEST( greeting == "hello" );
		HK_TEST( greeting != HK_NULL );
		HK_TEST( greeting != goodbye );
		HK_TEST( null  == HK_NULL );
	}

	{
		// simple
		testPathNormalize("../test.txt", "../test.txt" );
		testPathNormalize("..", ".." );
		testPathNormalize("", "" );
		testPathNormalize("test.txt", "test.txt" );

		// combinations of ..
		testPathNormalize("C:\\temp\\test\\..\\..\\testing.txt", "C:/testing.txt");
		testPathNormalize("C:\\temp\\test\\..\\../testing\\..\\more/testing\\test.txt", "C:/more/testing/test.txt" );
		testPathNormalize("a/b/../..", "" );

		// leading ..
		testPathNormalize("../foo/../bar/../test.txt", "../test.txt" );
		testPathNormalize("../foo/../bar/../../test.txt", "../../test.txt" );
		testPathNormalize("../../a/b", "../../a/b" );
		testPathNormalize("../a/..", ".." );
		testPathNormalize("../a/../..", "../.." );

		// absolute paths
		testPathNormalize("/a/b/../c", "/a/c");
		testPathNormalize("/", "/");
		testPathNormalize("/a/../..", "/");//? invalid, past root
		testPathNormalize("/a/..", "/");
		testPathNormalize("/a/b/..", "/a");

		// trailing slashes
		testPathNormalize("/a/b/../", "/a");

		// dots and leading dots
		testPathNormalize("./a/b/../", "a");
		testPathNormalize("./a/./b", "a/b");
		testPathNormalize("/a/./b", "/a/b");
		testPathNormalize("/a/./b/.", "/a/b");

		// double slashes? we want to preserve unc paths
		testPathNormalize("//foo/bar/.", "//foo/bar");
		testPathNormalize("\\\\foo\\bar", "//foo/bar");
	}

	{
		testPathAppend( "foo/bar",	"", "/foo", "/bar");
		testPathAppend( "/foo/bar",	"/foo", "/bar", "");
		testPathAppend( "foo/bar",	"", "foo", "/bar");
		testPathAppend( "foo/bar",	"", "foo", "bar");
		testPathAppend( "foo/b/ar",	"", "foo", "b/ar");
		testPathAppend( "",			"", "", "");
		testPathAppend( "a",		"", "", "a");
		testPathAppend( "a/b",		"", "/a", "b");
		testPathAppend( "a/b",		"", "/a", "b/");
		testPathAppend( "a/b",		"", "/a", "/b/");
		testPathAppend( "/root/a",	"/root", "", "a");
		testPathAppend( "/root/x/y","/root", "/x", "/y");
		testPathAppend( "root/x/y",	"root",  "x", "y");
		testPathAppend( "root/x/y",	"root",  "x/", "/y");
		testPathAppend( "root/x/y",	"root",  "x/", "/y/");
		testPathAppend( "root/y",	"root",  "/", "//y//");
	}

	{
		const char* fooBarBaz[] = {"foo", "bar", "baz", HK_NULL };
		testSplitEqual("foo/bar/baz", fooBarBaz);

		const char* empty[] = {"", HK_NULL };
		testSplitEqual("", empty);

		const char* empty2[] = {"", "", HK_NULL };
		testSplitEqual("/", empty2);

		const char* slashFoo[] = {"", "foo", HK_NULL };
		testSplitEqual("/foo", slashFoo);

		const char* fooSlash[] = {"foo", "", HK_NULL };
		testSplitEqual("foo/", fooSlash);

		const char* empty3[] = {"", "", "", HK_NULL };
		testSplitEqual("//", empty3);
	}

	{
		// checking hkStringPtr::setPointerAligned()

		HK_ALIGN(const char testStr[],2) = "Hello Test!";
		hkStringPtr str("Old Content");
		str.setPointerAligned(testStr);
		HK_TEST(str.cString() == testStr);

		str.setPointerAligned(HK_NULL);
		HK_TEST(str.cString() == HK_NULL);
		
		hkStringPtr str2("Hello");
		str.setPointerAligned(str2.cString());
		HK_TEST(str.cString() == str2.cString());

		HK_TEST_ASSERT(0x3d02cfc8, str.setPointerAligned(testStr+1));
	}

	string_memset16_test();
	string_int24w();
	hkarray_stringbuf_not_allowed();
	string_encoding_equal();
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(string_main, "Fast", "Common/Test/UnitTest/Base/", __FILE__);

/*
 * Havok SDK - Base file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
