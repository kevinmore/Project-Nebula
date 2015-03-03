/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/System/Io/FileSystem/hkFileSystem.h>
#include <Common/Base/System/Io/IStream/hkIStream.h>
#include <Common/Base/System/Io/Writer/hkStreamWriter.h>
#include <Common/Base/System/Io/Reader/hkStreamReader.h>
#include <Common/Base/System/Io/Reader/Buffered/hkBufferedStreamReader.h>

static const char filename[] = "testfile.txt";

class FakeStreamWriter : public hkStreamWriter
{
	public:

		static int numInstance;

		FakeStreamWriter()
		{
			numInstance++;
		}
		~FakeStreamWriter()
		{
			numInstance--;
		}

		hkBool isOk() const { return true; }
		int write(const void* buf, int nbytes) { return 0; }
		void flush() { }
};
int FakeStreamWriter::numInstance;

template <typename T>
void testRead( const char* inString, T shouldBe )
{
	hkIstream is(inString, hkString::strLen(inString));
	T intVal;
	is >> intVal;
	HK_TEST2(intVal == shouldBe, "Got "<<intVal<<" should be "<<shouldBe);
	HK_TEST(is.isOk());
}

namespace
{
	struct BlobStreamReader : public hkStreamReader
	{
		hkArray<const char*> m_bits;
		bool m_isOk;

		BlobStreamReader(const char* b0=HK_NULL, const char* b1=HK_NULL, const char* b2=HK_NULL)
			: m_isOk(true)
		{
			if( b2 ) { m_bits.pushBack(b2); }
			if( b1 ) { m_bits.pushBack(b1); }
			if( b0 ) { m_bits.pushBack(b0); }
		}

		virtual hkBool isOk() const { return m_isOk; }

		virtual int read(void* buf, int nb)
		{
			if( nb == 0 || m_isOk == false )
			{
				return 0;
			}
			if( m_bits.getSize() == 0 )
			{
				m_isOk = false;
				return false;
			}
			int l = hkString::strLen(m_bits.back());
			int n = hkMath::min2(l, nb);
			hkMemUtil::memCpy(buf, m_bits.back(), n);
			if( l == n )
			{
				m_bits.popBack(); // read the whole thing
			}
			else
			{
				m_bits.back() += n; // only read some
			}
			return n;
		}
	};

	struct BufferedStreamTester
	{
		BufferedStreamTester(const char* b0=HK_NULL, const char* b1=HK_NULL, const char* b2=HK_NULL)
			: m_blobs(b0,b1,b2)
			, m_buf(&m_blobs)
		{
		}

		hkBufferedStreamReader* operator->() { return &m_buf; }

		BlobStreamReader m_blobs;
		hkBufferedStreamReader m_buf;
	};
}

void buffer_read_test()
{
	char buf[1024] = {};
	{
		BufferedStreamTester r("hello", "world");
		HK_TEST( r->read(buf, 100) == 10 );
	}

	for( int i = 0; i < 10; ++i )
	{
		BufferedStreamTester r("hello", "world");
		HK_TEST2( r->peek(buf, i) == i, i );
		HK_TEST2( r->read(buf, i) == i, i );
		HK_TEST2( r->peek(buf, 100) == 10-i, i );
		HK_TEST2( r->read(buf, 100) == 10-i, i );
	}
	{
		BufferedStreamTester r("hello", "world");
		HK_TEST( r->peek(buf, 4) == 4 );
		HK_TEST( r->read(buf, 4) == 4 );
		HK_TEST( r->peek(buf, 100) == 6 );
		HK_TEST( r->read(buf, 100) == 6 );
	}
	
	for( int i = 0; i < 100; ++i )
	{
		BufferedStreamTester r("hello", "world");
		HK_TEST2( r->skip(i) == hkMath::min2(i,10), i );
	}
	{
		BufferedStreamTester r("hello", "world");
		HK_TEST( r->skip(100) == 10 );
	}
	{
		BufferedStreamTester r("hello", "world");
		for( int i = 0; i < 5; ++i )
		{ 
			HK_TEST( r->read(buf, 2) == 2 );
		}
	}
	{
		BufferedStreamTester r("hello", "world");
		for( int i = 0; i < 5; ++i )
		{ 
			HK_TEST( r->skip(2) == 2 );
		}
	}
	{
		BufferedStreamTester r("hello", "world");
		// Test the refill correctly moves existing data in the buffer
		char peekbuf[20];
		HK_TEST( r->peek(peekbuf, 4) == 4 );
		HK_TEST( r->read(buf, 2) == 2);
		HK_TEST(buf[0] == 'h');
		HK_TEST(buf[1] == 'e');
		HK_TEST(r->peek(peekbuf, 15) == 8);
		HK_TEST(r->read(buf, 2) == 2);
		HK_TEST(buf[0] == 'l');
		HK_TEST(buf[1] == 'l');
		HK_TEST(r->read(buf, 2) == 2);
		HK_TEST(buf[0] == 'o');
		HK_TEST(buf[1] == 'w');
		HK_TEST(r->peek(peekbuf, 10) == 4);
	}
}

void istream_test()
{
	{
		/*
		FakeStreamWriter* f;
		{
			f = new FakeStreamWriter();
			hkOstream somestream(f);
			f->removeReference();
		}
		HK_TEST(FakeStreamWriter::numInstance==0);

		{
			f = new FakeStreamWriter();
			hkOstream somestream(f);
		}
		HK_TEST(FakeStreamWriter::numInstance==1);
		f->removeReference();
		HK_TEST(FakeStreamWriter::numInstance==0);
		*/
	}


	{
		testRead<int>("\n0x582\n", 0x582);
		testRead<int>("34", 34);
		testRead<int>("35 ", 35);
		testRead<int>("-1 ", -1);
		testRead<unsigned>("-1 ", unsigned(-1));
		testRead<unsigned>("0x00f ", 0xf);
		testRead<unsigned>("0712", 0712);
		testRead<hkInt32>("0xffffffff", -1);
		testRead<hkInt32>("4294967295", -1);
		testRead<hkInt32>("2147483647", hkUint32(-1)>>1);
		testRead<hkInt32>("-2147483647", -2147483647);
		testRead<hkUint64>("0xffffffffffffffff", hkUint64(-1) );
		testRead<hkInt64>("0xffffffffffffffff", -1 );
	}
	{
		/*
		hkOstream os(filename);

		if( HK_TEST( os.isOk() ) )
		{
			hkString sval("33 45.5 hello");
			os << sval;
			HK_TEST(os.isOk());
		}
		*/
	}

	{
		/*
		hkIstream is(filename);

		if( HK_TEST( is.isOk() ) )
		{
			int ival;
			is >> ival;
			HK_TEST(ival==33);
			HK_TEST(is.isOk());

			float fval;
			is >> fval;
						
			HK_TEST(fval==45.5);
			HK_TEST(is.isOk());

			hkString sval;
			is >> sval;
			HK_TEST(!is.isOk());	// Should have read to EOF
		}
		*/
	}

	{
		/*
		hkOstream os(filename);

		if( HK_TEST( os.isOk() ) )
		{
			hkString sval("Testing ");
			os << sval;
			HK_TEST(os.isOk());
		}
		*/
	}
	{
		hkArray<char> cbuf;
		hkOstream os(cbuf);

		HK_TEST( cbuf.getCapacity() >= 1 );
		HK_TEST( cbuf.begin()[0] == 0 );
		const char* testString = "hello world";
		os << testString;
		HK_TEST( hkString::strCmp( cbuf.begin(), testString) == 0 );
		HK_TEST( hkString::strLen( cbuf.begin() ) == hkString::strLen( testString ) );

		HK_TEST( os.getStreamWriter()->seek( 6, hkStreamWriter::STREAM_SET ) == HK_SUCCESS );
		os << "WORLD WORLD";
		HK_TEST( cbuf.begin()[ cbuf.getSize() ] == 0 );
	}
}


void streamcreate_test()
{
// 	hkIstream is0("testfile.txt");
// 	HK_TEST(is0.isOk());
// 	hkIstream is1("nonexistantfile.txt");
// 	HK_TEST(is1.isOk()==false);
}


//
// Tests archives
//
void archive_test()
{
	//
	// tests the hkIArchive and hkOArchive classes for binary input and output
	// In this case, file streams are created
	// first, some test data of each supported type is written to file.
	// Then the file is read back in and compared to the original.
	//
	{
		// output
		{
			/*
			hkOfArchive oa("testbindata.txt");
			
			// test data
			char c = 'a';
			unsigned char uc = 25;
			short s = -12345;
			unsigned short us = 12345;
			int i = -12345;
			unsigned int ui = 12345;
			long int li = -12345;
			unsigned long int uli = 12345;
			float f = -1.2345f;
			double d = -1.2345;

			// write
			oa.write8(c);
			
			oa.write8u(uc);
			
			oa.write16(s);
			
			oa.write16u(us);

			oa.write32(i);
			
			oa.write32u(ui);

			oa.write64(li);
			
			oa.write64u(uli);

			oa.writeFloat32(f);
			
			oa.writeDouble64(d);
			*/
		}
		
		// input
		{
			/*
			hkIfArchive ia("testbindata.txt");

			char c = ia.read8();
			HK_TEST(c == 'a');
	
			unsigned char uc = ia.read8u();
			HK_TEST(uc == 25);

			short s = ia.read16();
			HK_TEST(s == -12345);

			unsigned short us = ia.read16u();
			HK_TEST(us == 12345);

			int i = ia.read32();
			HK_TEST(i == -12345);

			unsigned int ui = ia.read32u();
			HK_TEST(ui == 12345);

			hkInt64 li = ia.read64();
			HK_TEST(li == -12345);

			hkUint64 uli = ia.read64u();
			HK_TEST(uli == 12345);

			float f = ia.readFloat32();
			HK_TEST(f == -1.2345f);

			double d = ia.readDouble64();
			HK_TEST(d == -1.2345);
			*/
		}
	}
}


static int stream_main()
{
	buffer_read_test();
	istream_test();
	streamcreate_test();
	archive_test();

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(stream_main, "Fast", "Common/Test/UnitTest/Base/", __FILE__     );

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
