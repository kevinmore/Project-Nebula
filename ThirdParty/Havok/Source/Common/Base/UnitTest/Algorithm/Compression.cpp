/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>

#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Algorithm/Compression/hkBufferCompression.h>
#include <Common/Base/Algorithm/Compression/hkCompression.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>
#include <Common/Base/System/Io/IStream/hkIStream.h>
#include <Common/Base/System/Io/Reader/Compressed/hkCompressedStreamReader.h>
#include <Common/Base/System/Io/Reader/Memory/hkMemoryStreamReader.h>
#include <Common/Base/System/Io/Writer/Buffered/hkBufferedStreamWriter.h>
#include <Common/Base/System/Io/Writer/Compressed/hkCompressedStreamWriter.h>
#include <Common/Base/System/Stopwatch/hkStopwatch.h>
#include <Common/Serialize/Util/hkStructureLayout.h>
#include <Common/Base/Fwd/hkcstdio.h>

// This file contains some code to benchmark the compression routines against
// third-party open-source compression (LZOP, LZMA and Zlib). Due to license
// issues, these cannot be distributed with Havok, so those benchmarks are
// internal-only
// #define BENCHMARK_THIRDPARTY_COMPRESSION

enum { BUFSIZE = 128 * 1024 };


#ifdef BENCHMARK_THIRDPARTY_COMPRESSION
namespace LZMA
{
	int compress(char* in, int inlen, char* out, int outlen, bool fastest);
	int decompress(char* in, int inlen, char* out, int outlen);
}
namespace ZLib
{
	int compress(char* in, int inlen, char* out, int outlen, bool fastest);
	int decompress(char* in, int inlen, char* out, int outlen);

}
namespace LZOP
{
	int compress(char* in, int inlen, char* out, int outlen_, bool fastest);
	int decompress(char* in, int inlen, char* out, int outlen_);
}
#endif


struct compressor
{
	typedef int (HK_CALL *compress_f)(const void* in, int inlen, void* out, int outlen, bool fastest);
	typedef int (HK_CALL *decompress_f)(const void* in, int inlen, void* out, int outlen);

	compress_f compress;
	decompress_f decompress;
	const char* name;
	bool fastest;

	compressor(compress_f c, decompress_f d, const char* n, bool fast)
		: compress(c)
		, decompress(d)
		, name(n)
		, fastest(fast)
	{
	}

	void test_comp(const void* data, int dataSize, const char* dataSource = "")
	{
		hkArray<char>::Temp compressedBuf(BUFSIZE);
		hkArray<char>::Temp uncompressedBuf(BUFSIZE);
		void* compressed = compressedBuf.begin();
		void* uncompressed = uncompressedBuf.begin();

		int COMP_ITERS = hkMath::max2(1, 1 * 1024 * 1024 / dataSize), DECOMP_ITERS=hkMath::max2(1, 10 * 1024 * 1024 / dataSize);
		hkStopwatch compression_timer, decompression_timer;
		int compressedSize=0;

		compression_timer.start();
		for (int i=0;i<COMP_ITERS;i++)
		{
			compressedSize = compress(data, dataSize, compressed, BUFSIZE, fastest);
		}
		compression_timer.stop();
		HK_TEST(compressedSize!=0);

		int deCompressedSize = 0;
		decompression_timer.start();
		for (int i=0;i<DECOMP_ITERS;i++)
		{
			deCompressedSize = decompress(compressed, compressedSize, uncompressed, BUFSIZE);
		}
		decompression_timer.stop();
		HK_TEST(deCompressedSize!=0);
		HK_TEST(deCompressedSize == dataSize);
		HK_TEST(hkString::memCmp(data, uncompressed, dataSize)==0);

// 		printf("%s: [%s]\n\t%2d%% %8d %8d (%3.1f MB/s - %3.1f MB/s)\n",
// 			name, dataSource,
// 			int(100*hkReal(compressedSize)/dataSize), compressedSize, dataSize,
// 			(hkReal(dataSize) / 1024.0 / 1024.0) / compression_timer.getElapsedSeconds() * COMP_ITERS,
// 			(hkReal(dataSize) / 1024.0 / 1024.0) / decompression_timer.getElapsedSeconds() * DECOMP_ITERS);
	}
};

static const char* dataset[] =
{
	"Resources/Animation/ShowCase/Animations/hkGetupBack1.hkt",
	"Resources/Animation/HavokGirl/hkIdle_%s",
	"Resources/Animation/FireFighter/Animations/hkRunJump.hkt",
	"Resources/Behavior/GameAssets/Orcs/Project/Behaviors/Troll.hkt",
	"Resources/Cloth/Bootstrap/Troll.hkt",
	"Resources/Cloth/female/female_pris_cloth_setup_full.hkt",
	"Resources/Cloth/male/animations/male_jumpup.hkt",
	"Resources/Physics2012/Objects/Teapot.xml",
	"Resources/Physics2012/Concaves/house_%s",
};

static int HK_CALL mycompress(const void* in, int inlen, void* out, int outlen, bool fastest)
{
	return (int)hkBufferCompression::compressBuffer(in, inlen, out, outlen, fastest);
}
static int HK_CALL mydecompress(const void* in, int inlen, void* out_start, int outlen)
{
	void* out = out_start;
	hkBufferCompression::decompressBufferFast(in, inlen, out);
	int size = hkGetByteOffsetInt(out_start, out);
	return size;
}

compressor algos[] = 
{
	#ifdef BENCHMARK_THIRDPARTY_COMPRESSION
		compressor(LZMA::compress, LZMA::decompress, "LZMA", false),
		compressor(ZLib::compress, ZLib::decompress, "Zlib -1", true),
		compressor(ZLib::compress, ZLib::decompress, "Zlib -9", false),
		compressor(LZOP::compress, LZOP::decompress, "LZOP", false),
	#endif

	compressor(mycompress, mydecompress, "hkCompression [hash]", true),
	compressor(mycompress, mydecompress, "hkCompression [suffixtree]", false),
};

// static void test_all_algos(int n)
// {
// 	for (int i=0;i<sizeof(algos)/sizeof(algos[0]);i++)
// 		algos[i].test_comp(n);
// 	printf("\n");
// }


static void test_dataset()
{
	hkArray<char>::Temp data; data.setSize(BUFSIZE);
	hkStringBuf hkxSuffix;
	{
		const hkStructureLayout::LayoutRules& rules = hkStructureLayout::HostLayoutRules;
		hkxSuffix.printf("L%d%d%d%d.hkx", 
			rules.m_bytesInPointer,
			rules.m_littleEndian? 1 : 0,
			rules.m_reusePaddingOptimization? 1 : 0,
			rules.m_emptyBaseClassOptimization? 1 : 0);
	}
	
	for( int i=0; i < (int)HK_COUNT_OF(dataset); i++ )
	{
		hkStringBuf filename; filename.printf(dataset[i], hkxSuffix.cString());
		hkIstream d(filename);
		if( d.isOk() )
		{
			int rawSize = d.read(data.begin(), BUFSIZE);
			for (int j = 0; j < (int)HK_COUNT_OF(algos); j++)
			{
				algos[j].test_comp(data.begin(), rawSize, filename.cString());

				hkArray<char> compBuf;
				{
					hkOstream out(compBuf);
					hkCompressedStreamWriter writer(out.getStreamWriter());
					writer.write( data.begin(), rawSize );
					writer.flush();
				}
				{
					hkMemoryStreamReader memread(compBuf.begin(), compBuf.getSize(), hkMemoryStreamReader::MEMORY_INPLACE);
					hkCompressedStreamReader reader(&memread);

					hkArray<char>::Temp rawBuf(BUFSIZE);
					int nread = reader.read( rawBuf.begin(), rawSize );
					HK_TEST( nread == rawSize );
					char buf[10];
					HK_TEST( reader.read(buf, 1) == 0 ); //eof
					HK_TEST( hkString::memCmp(rawBuf.begin(), data.begin(), rawSize) == 0 );
				}
			}
		}
		else
		{
			HK_WARN(0xfa3c3500, "Couldn't read '" << filename.cString() << "', does it exist?");
		}
	}

	hkPseudoRandomGenerator g(4321910);
	for( int i=0; i < (int)HK_COUNT_OF(dataset); i++ )
	{
		hkStringBuf filename; filename.printf(dataset[i], hkxSuffix.cString());
		hkIstream d(filename);
		if( d.isOk() )
		{
			int rawSize = d.read(data.begin(), BUFSIZE);
			for (int j = 0; j < (int)HK_COUNT_OF(algos); j++)
			{
				hkArray<char> compBuf;
				{
					hkOstream out(compBuf);
					hkCompressedStreamWriter writer(out.getStreamWriter());
					int written = 0;
					while (written < rawSize)
					{
						int nbytes = hkMath::min2(g.getRand32() % 3000 + 1, rawSize - written);
						writer.write( data.begin() + written, nbytes );
						written += nbytes;
						if (g.getRand32() % 500 == 0) writer.flush();
					}
					
					writer.flush();
				}
				{
					hkMemoryStreamReader memread(compBuf.begin(), compBuf.getSize(), hkMemoryStreamReader::MEMORY_INPLACE);
					hkCompressedStreamReader reader(&memread);

					hkArray<char>::Temp rawBuf(BUFSIZE);
					int nread = 0;
					while (1)
					{
						int nbytes = (g.getRand32() % 3000) + 1;
						int r = reader.read( rawBuf.begin() + nread, nbytes );
						HK_TEST( hkString::memCmp(rawBuf.begin() + nread, data.begin() + nread, r) == 0);
						nread += r;
						HK_ASSERT(0x4325BACD, r == nbytes || nread == rawSize);
						if (!r)
						{
							break;
						}
					}
					
					HK_TEST( nread == rawSize );
					char buf[10];
					HK_TEST( reader.read(buf, 1) == 0 ); //eof
					HK_TEST( hkString::memCmp(rawBuf.begin(), data.begin(), rawSize) == 0 );
				}
			}
		}
		else
		{
			HK_WARN(0xfa3c3501, "Couldn't read '" << filename.cString() << "', does it exist?");
		}
	}
}

static void test_comp(const void* data, int len, int bsz = 1 << 16)
{
	hkArray<char>::Temp compressedBuf(BUFSIZE);
	hkArray<char>::Temp uncompressedBuf(BUFSIZE);
	void* compressed = compressedBuf.begin();
	void* uncompressed = uncompressedBuf.begin();

	const int COMP_ITERS = 1;//hkMath::max2(1, 1 * 1024 * 1024 / len);
	hkStopwatch compression_timer;
	int compSize = -1;
	{
		hkString::memSet(compressedBuf.begin(), 0xfe,  compressedBuf.getSize());
		const void* pdata=0;
		void* pcomp=0;
		hkCompression::Result res=hkCompression::COMP_ERROR;
		compression_timer.start();
		for (int i=0; i<COMP_ITERS; i++)
		{
			pdata = data;
			pcomp = compressed;
			res = hkCompression::compress(pdata, len, pcomp, BUFSIZE, bsz);
		}
		compression_timer.stop();
		
		HK_TEST(res == hkCompression::COMP_NEEDINPUT);
		HK_TEST(pdata == hkAddByteOffsetConst(data,len) );
		compSize = hkGetByteOffsetInt(compressed, pcomp);
		HK_TEST( hkUchar(compressedBuf[compSize]) == 0xfe);
	}
	
	const int DECOMP_ITERS=hkMath::max2(1, 10 * 1024 * 1024 / len);
	hkStopwatch decompression_timer;

	{
		hkString::memSet(uncompressedBuf.begin(), 0xfe, uncompressedBuf.getSize());
		const void* pcomp2=0;
		void* ucomp=0;
		hkCompression::Result res=hkCompression::COMP_ERROR;
		decompression_timer.start();
		for (int i=0; i<DECOMP_ITERS; i++)
		{
			pcomp2 = compressed;
			ucomp = uncompressed;
			res	= hkCompression::decompress(pcomp2, compSize, ucomp, BUFSIZE, true);
		}
		decompression_timer.stop();
		HK_TEST( res == hkCompression::COMP_NEEDINPUT );
		HK_TEST( ucomp == hkAddByteOffset(uncompressed, len) );
		HK_TEST( hkString::memCmp(data, uncompressed, len) == 0 );
		HK_TEST( hkUchar(uncompressedBuf[len]) == 0xfe );
	}


// 	printf("%.1f%% of original size (%.1f MB/s compression, %.1f MB/s decompression, blocksize %d) %d %d %f %f\n", 
// 		100 * float(compSize) / len,
// 		(hkReal(len) / 1024.0 / 1024.0) / compression_timer.getElapsedSeconds() * COMP_ITERS,
// 		(hkReal(len) / 1024.0 / 1024.0) / decompression_timer.getElapsedSeconds() * DECOMP_ITERS,
// 		bsz,
// 		COMP_ITERS, DECOMP_ITERS, compression_timer.getElapsedSeconds(),decompression_timer.getElapsedSeconds());
}

static void test_xmldata()
{
	static const char xmldata[] = "<?xml version=\"1.0\" encoding=\"ascii\"?>"
		"<hkpackfile classversion=\"7\" contentsversion=\"hk_2010.1.0-$$\" toplevelobject=\"#0001\">"
		""
		"	<hksection name=\"__types__\">"
		""
		"		<hkobject name=\"#0004\" class=\"hkClass\" signature=\"0x14425e51\">"
		"			<hkparam name=\"name\">hkBaseObject</hkparam>"
		"			<hkparam name=\"parent\">null</hkparam>"
		"			<hkparam name=\"objectSize\">4</hkparam>"
		"			<hkparam name=\"numImplementedInterfaces\">1</hkparam>"
		"			<hkparam name=\"declaredEnums\" numelements=\"0\"></hkparam>"
		"			<hkparam name=\"declaredMembers\" numelements=\"0\"></hkparam>"
		"			<!-- defaults SERIALIZE_IGNORED -->"
		"			<!-- attributes SERIALIZE_IGNORED -->"
		"			<hkparam name=\"flags\">0</hkparam>"
		"			<hkparam name=\"describedVersion\">0</hkparam>"
		"		</hkobject>"
		""
		"		<hkobject name=\"#0003\" class=\"hkClass\" signature=\"0x14425e51\">"
		"			<hkparam name=\"name\">hkReferencedObject</hkparam>"
		"			<hkparam name=\"parent\">#0004</hkparam>"
		"			<hkparam name=\"objectSize\">8</hkparam>"
		"			<hkparam name=\"numImplementedInterfaces\">0</hkparam>"
		"			<hkparam name=\"declaredEnums\" numelements=\"0\"></hkparam>"
		"			<hkparam name=\"declaredMembers\" numelements=\"2\">"
		"				<hkobject>"
		"					<hkparam name=\"name\">memSizeAndFlags</hkparam>"
		"					<hkparam name=\"class\">null</hkparam>"
		"					<hkparam name=\"enum\">null</hkparam>"
		"					<hkparam name=\"type\">TYPE_UINT16</hkparam>"
		"					<hkparam name=\"subtype\">TYPE_VOID</hkparam>"
		"					<hkparam name=\"cArraySize\">0</hkparam>"
		"					<hkparam name=\"flags\">SERIALIZE_IGNORED</hkparam>"
		"					<hkparam name=\"offset\">4</hkparam>"
		"					<!-- attributes SERIALIZE_IGNORED -->"
		"				</hkobject>"
		"				<hkobject>"
		"					<hkparam name=\"name\">referenceCount</hkparam>"
		"					<hkparam name=\"class\">null</hkparam>"
		"					<hkparam name=\"enum\">null</hkparam>"
		"					<hkparam name=\"type\">TYPE_INT16</hkparam>"
		"					<hkparam name=\"subtype\">TYPE_VOID</hkparam>"
		"					<hkparam name=\"cArraySize\">0</hkparam>"
		"					<hkparam name=\"flags\">SERIALIZE_IGNORED</hkparam>"
		"					<hkparam name=\"offset\">6</hkparam>"
		"					<!-- attributes SERIALIZE_IGNORED -->"
		"				</hkobject>"
		"			</hkparam>"
		"			<!-- defaults SERIALIZE_IGNORED -->"
		"			<!-- attributes SERIALIZE_IGNORED -->"
		"			<hkparam name=\"flags\">0</hkparam>"
		"			<hkparam name=\"describedVersion\">0</hkparam>"
		"		</hkobject>"
		""
		"		<hkobject name=\"#0002\" class=\"hkClass\" signature=\"0x14425e51\">"
		"			<hkparam name=\"name\">hkRootLevelContainerNamedVariant</hkparam>"
		"			<hkparam name=\"parent\">null</hkparam>"
		"			<hkparam name=\"objectSize\">12</hkparam>"
		"			<hkparam name=\"numImplementedInterfaces\">0</hkparam>"
		"			<hkparam name=\"declaredEnums\" numelements=\"0\"></hkparam>"
		"			<hkparam name=\"declaredMembers\" numelements=\"3\">"
		"				<hkobject>"
		"					<hkparam name=\"name\">name</hkparam>"
		"					<hkparam name=\"class\">null</hkparam>"
		"					<hkparam name=\"enum\">null</hkparam>"
		"					<hkparam name=\"type\">TYPE_STRINGPTR</hkparam>"
		"					<hkparam name=\"subtype\">TYPE_VOID</hkparam>"
		"					<hkparam name=\"cArraySize\">0</hkparam>"
		"					<hkparam name=\"flags\">0</hkparam>"
		"					<hkparam name=\"offset\">0</hkparam>"
		"					<!-- attributes SERIALIZE_IGNORED -->"
		"				</hkobject>"
		"				<hkobject>"
		"					<hkparam name=\"name\">className</hkparam>"
		"					<hkparam name=\"class\">null</hkparam>"
		"					<hkparam name=\"enum\">null</hkparam>"
		"					<hkparam name=\"type\">TYPE_STRINGPTR</hkparam>"
		"					<hkparam name=\"subtype\">TYPE_VOID</hkparam>"
		"					<hkparam name=\"cArraySize\">0</hkparam>"
		"					<hkparam name=\"flags\">0</hkparam>"
		"					<hkparam name=\"offset\">4</hkparam>"
		"					<!-- attributes SERIALIZE_IGNORED -->"
		"				</hkobject>"
		"				<hkobject>"
		"					<hkparam name=\"name\">variant</hkparam>"
		"					<hkparam name=\"class\">#0003</hkparam>"
		"					<hkparam name=\"enum\">null</hkparam>"
		"					<hkparam name=\"type\">TYPE_POINTER</hkparam>"
		"					<hkparam name=\"subtype\">TYPE_STRUCT</hkparam>"
		"					<hkparam name=\"cArraySize\">0</hkparam>"
		"					<hkparam name=\"flags\">0</hkparam>"
		"					<hkparam name=\"offset\">8</hkparam>"
		"					<!-- attributes SERIALIZE_IGNORED -->"
		"				</hkobject>"
		"			</hkparam>"
		"			<!-- defaults SERIALIZE_IGNORED -->"
		"			<!-- attributes SERIALIZE_IGNORED -->"
		"			<hkparam name=\"flags\">0</hkparam>"
		"			<hkparam name=\"describedVersion\">0</hkparam>"
		"		</hkobject>"
		""
		"		<hkobject name=\"#0005\" class=\"hkClass\" signature=\"0x14425e51\">"
		"			<hkparam name=\"name\">hkRootLevelContainer</hkparam>"
		"			<hkparam name=\"parent\">null</hkparam>"
		"			<hkparam name=\"objectSize\">12</hkparam>"
		"			<hkparam name=\"numImplementedInterfaces\">0</hkparam>"
		"			<hkparam name=\"declaredEnums\" numelements=\"0\"></hkparam>"
		"			<hkparam name=\"declaredMembers\" numelements=\"1\">"
		"				<hkobject>"
		"					<hkparam name=\"name\">namedVariants</hkparam>"
		"					<hkparam name=\"class\">#0002</hkparam>"
		"					<hkparam name=\"enum\">null</hkparam>"
		"					<hkparam name=\"type\">TYPE_ARRAY</hkparam>"
		"					<hkparam name=\"subtype\">TYPE_STRUCT</hkparam>"
		"					<hkparam name=\"cArraySize\">0</hkparam>"
		"					<hkparam name=\"flags\">0</hkparam>"
		"					<hkparam name=\"offset\">0</hkparam>"
		"					<!-- attributes SERIALIZE_IGNORED -->"
		"				</hkobject>"
		"			</hkparam>"
		"			<!-- defaults SERIALIZE_IGNORED -->"
		"			<!-- attributes SERIALIZE_IGNORED -->"
		"			<hkparam name=\"flags\">0</hkparam>"
		"			<hkparam name=\"describedVersion\">0</hkparam>"
		"		</hkobject>"
		""
		"		<hkobject name=\"#0008\" class=\"hkClass\" signature=\"0x14425e51\">"
		"			<hkparam name=\"name\">hclAction</hkparam>"
		"			<hkparam name=\"parent\">#0003</hkparam>"
		"			<hkparam name=\"objectSize\">8</hkparam>"
		"			<hkparam name=\"numImplementedInterfaces\">0</hkparam>"
		"			<hkparam name=\"declaredEnums\" numelements=\"0\"></hkparam>"
		"			<hkparam name=\"declaredMembers\" numelements=\"0\"></hkparam>"
		"			<!-- defaults SERIALIZE_IGNORED -->"
		"			<!-- attributes SERIALIZE_IGNORED -->"
		"			<hkparam name=\"flags\">0</hkparam>"
		"			<hkparam name=\"describedVersion\">0</hkparam>"
		"		</hkobject>"
		""
		"		<hkobject name=\"#0011\" class=\"hkClass\" signature=\"0x14425e51\">"
		"			<hkparam name=\"name\">hclBufferUsage</hkparam>"
		"			<hkparam name=\"parent\">null</hkparam>"
		"			<hkparam name=\"objectSize\">5</hkparam>"
		"			<hkparam name=\"numImplementedInterfaces\">0</hkparam>"
		"			<hkparam name=\"declaredEnums\" numelements=\"2\">"
		"				<hkobject>"
		"					<hkparam name=\"name\">Component</hkparam>"
		"					<hkparam name=\"items\" numelements=\"4\">"
		"						<hkobject>"
		"							<hkparam name=\"value\">0</hkparam>"
		"							<hkparam name=\"name\">COMPONENT_POSITION</hkparam>"
		"						</hkobject>"
		"						<hkobject>"
		"							<hkparam name=\"value\">1</hkparam>"
		"							<hkparam name=\"name\">COMPONENT_NORMAL</hkparam>"
		"						</hkobject>"
		"						<hkobject>"
		"							<hkparam name=\"value\">2</hkparam>"
		"							<hkparam name=\"name\">COMPONENT_TANGENT</hkparam>"
		"						</hkobject>"
		"						<hkobject>"
		"							<hkparam name=\"value\">3</hkparam>"
		"							<hkparam name=\"name\">COMPONENT_BITANGENT</hkparam>"
		"						</hkobject>"
		"					</hkparam>"
		"					<!-- attributes SERIALIZE_IGNORED -->"
		"					<hkparam name=\"flags\">0</hkparam>"
		"				</hkobject>"
		"				<hkobject>"
		"					<hkparam name=\"name\">InternalFlags</hkparam>"
		"					<hkparam name=\"items\" numelements=\"5\">"
		"						<hkobject>"
		"							<hkparam name=\"value\">0</hkparam>"
		"							<hkparam name=\"name\">USAGE_NONE</hkparam>"
		"						</hkobject>"
		"						<hkobject>"
		"							<hkparam name=\"value\">1</hkparam>"
		"							<hkparam name=\"name\">USAGE_READ</hkparam>"
		"						</hkobject>"
		"						<hkobject>"
		"							<hkparam name=\"value\">2</hkparam>"
		"							<hkparam name=\"name\">USAGE_WRITE</hkparam>"
		"						</hkobject>"
		"						<hkobject>"
		"							<hkparam name=\"value\">4</hkparam>"
		"							<hkparam name=\"name\">USAGE_FULL_WRITE</hkparam>"
		"						</hkobject>"
		"						<hkobject>"
		"							<hkparam name=\"value\">8</hkparam>"
		"							<hkparam name=\"name\">USAGE_READ_BEFORE_WRITE</hkparam>"
		"						</hkobject>"
		"					</hkparam>"
		"					<!-- attributes SERIALIZE_IGNORED -->"
		"					<hkparam name=\"flags\">0</hkparam>"
		"				</hkobject>"
		"			</hkparam>"
		"			<hkparam name=\"declaredMembers\" numelements=\"2\">"
		"				<hkobject>"
		"					<hkparam name=\"name\">perComponentFlags</hkparam>"
		"					<hkparam name=\"class\">null</hkparam>"
		"					<hkparam name=\"enum\">null</hkparam>"
		"					<hkparam name=\"type\">TYPE_UINT8</hkparam>"
		"					<hkparam name=\"subtype\">TYPE_VOID</hkparam>"
		"					<hkparam name=\"cArraySize\">4</hkparam>"
		"					<hkparam name=\"flags\">0</hkparam>"
		"					<hkparam name=\"offset\">0</hkparam>"
		"					<!-- attributes SERIALIZE_IGNORED -->"
		"				</hkobject>"
		"				<hkobject>"
		"					<hkparam name=\"name\">trianglesRead</hkparam>"
		"					<hkparam name=\"class\">null</hkparam>"
		"					<hkparam name=\"enum\">null</hkparam>"
		"					<hkparam name=\"type\">TYPE_BOOL</hkparam>"
		"					<hkparam name=\"subtype\">TYPE_VOID</hkparam>"
		"					<hkparam name=\"cArraySize\">0</hkparam>"
		"					<hkparam name=\"flags\">0</hkparam>"
		"					<hkparam name=\"offset\">4</hkparam>"
		"					<!-- attributes SERIALIZE_IGNORED -->"
		"				</hkobject>"
		"			</hkparam>"
		"			<!-- defaults SERIALIZE_IGNORED -->"
		"			<!-- attributes SERIALIZE_IGNORED -->"
		"			<hkparam name=\"flags\">0</hkparam>"
		"			<hkparam name=\"describedVersion\">0</hkparam>"
		"		</hkobject>"
		""
		"		<hkobject name=\"#0010\" class=\"hkClass\" signature=\"0x14425e51\">"
		"			<hkparam name=\"name\">hclClothStateBufferAccess</hkparam>"
		"			<hkparam name=\"parent\">null</hkparam>"
		"			<hkparam name=\"objectSize\">16</hkparam>"
		"			<hkparam name=\"numImplementedInterfaces\">0</hkparam>"
		"			<hkparam name=\"declaredEnums\" numelements=\"0\"></hkparam>"
		"			<hkparam name=\"declaredMembers\" numelements=\"3\">"
		"				<hkobject>"
		"					<hkparam name=\"name\">bufferIndex</hkparam>"
		"					<hkparam name=\"class\">null</hkparam>"
		"					<hkparam name=\"enum\">null</hkparam>"
		"					<hkparam name=\"type\">TYPE_UINT32</hkparam>"
		"					<hkparam name=\"subtype\">TYPE_VOID</hkparam>"
		"					<hkparam name=\"cArraySize\">0</hkparam>"
		"					<hkparam name=\"flags\">0</hkparam>"
		"					<hkparam name=\"offset\">0</hkparam>"
		"					<!-- attributes SERIALIZE_IGNORED -->"
		"				</hkobject>"
		"				<hkobject>"
		"					<hkparam name=\"name\">bufferUsage</hkparam>"
		"					<hkparam name=\"class\">#0011</hkparam>"
		"					<hkparam name=\"enum\">null</hkparam>"
		"					<hkparam name=\"type\">TYPE_STRUCT</hkparam>"
		"					<hkparam name=\"subtype\">TYPE_VOID</hkparam>"
		"					<hkparam name=\"cArraySize\">0</hkparam>"
		"					<hkparam name=\"flags\">0</hkparam>"
		"					<hkparam name=\"offset\">4</hkparam>"
		"					<!-- attributes SERIALIZE_IGNORED -->"
		"				</hkobject>"
		"				<hkobject>"
		"					<hkparam name=\"name\">shadowBufferIndex</hkparam>"
		"					<hkparam name=\"class\">null</hkparam>"
		"					<hkparam name=\"enum\">null</hkparam>"
		"					<hkparam name=\"type\">TYPE_UINT32</hkparam>"
		"					<hkparam name=\"subtype\">TYPE_VOID</hkparam>"
		"					<hkparam name=\"cArraySize\">0</hkparam>"
		"					<hkparam name=\"flags\">0</hkparam>"
		"					<hkparam name=\"offset\">12</hkparam>"
		"					<!-- attributes SERIALIZE_IGNORED -->"
		"				</hkobject>"
		"			</hkparam>"
		"			<!-- defaults SERIALIZE_IGNORED -->"
		"			<!-- attributes SERIALIZE_IGNORED -->"
		"			<hkparam name=\"flags\">0</hkparam>"
		"			<hkparam name=\"describedVersion\">2</hkparam>"
		"		</hkobject>"
		""
		"		<hkobject name=\"#0009\" class=\"hkClass\" signature=\"0x14425e51\">"
		"			<hkparam name=\"name\">hclClothState</hkparam>"
		"			<hkparam name=\"parent\">#0003</hkparam>"
		"			<hkparam name=\"objectSize\">60</hkparam>"
		"			<hkparam name=\"numImplementedInterfaces\">0</hkparam>"
		"			<hkparam name=\"declaredEnums\" numelements=\"0\"></hkparam>"
		"			<hkparam name=\"declaredMembers\" numelements=\"5\">"
		"				<hkobject>"
		"					<hkparam name=\"name\">name</hkparam>"
		"					<hkparam name=\"class\">null</hkparam>"
		"					<hkparam name=\"enum\">null</hkparam>"
		"					<hkparam name=\"type\">TYPE_STRINGPTR</hkparam>"
		"					<hkparam name=\"subtype\">TYPE_VOID</hkparam>"
		"					<hkparam name=\"cArraySize\">0</hkparam>"
		"					<hkparam name=\"flags\">0</hkparam>"
		"					<hkparam name=\"offset\">8</hkparam>"
		"					<!-- attributes SERIALIZE_IGNORED -->"
		"				</hkobject>"
		"				<hkobject>"
		"					<hkparam name=\"name\">operators</hkparam>"
		"					<hkparam name=\"class\">null</hkparam>"
		"					<hkparam name=\"enum\">null</hkparam>"
		"					<hkparam name=\"type\">TYPE_ARRAY</hkparam>"
		"					<hkparam name=\"subtype\">TYPE_UINT32</hkparam>"
		"					<hkparam name=\"cArraySize\">0</hkparam>"
		"					<hkparam name=\"flags\">0</hkparam>"
		"					<hkparam name=\"offset\">12</hkparam>"
		"					<!-- attributes SERIALIZE_IGNORED -->"
		"				</hkobject>"
		"				<hkobject>"
		"					<hkparam name=\"name\">usedBuffers</hkparam>"
		"					<hkparam name=\"class\">#0010</hkparam>"
		"					<hkparam name=\"enum\">null</hkparam>"
		"					<hkparam name=\"type\">TYPE_ARRAY</hkparam>"
		"					<hkparam name=\"subtype\">TYPE_STRUCT</hkparam>"
		"					<hkparam name=\"cArraySize\">0</hkparam>"
		"					<hkparam name=\"flags\">0</hkparam>"
		"					<hkparam name=\"offset\">24</hkparam>"
		"					<!-- attributes SERIALIZE_IGNORED -->"
		"				</hkobject>"
		"				<hkobject>"
		"					<hkparam name=\"name\">usedTransformSets</hkparam>"
		"					<hkparam name=\"class\">null</hkparam>"
		"					<hkparam name=\"enum\">null</hkparam>"
		"					<hkparam name=\"type\">TYPE_ARRAY</hkparam>"
		"					<hkparam name=\"subtype\">TYPE_UINT32</hkparam>"
		"					<hkparam name=\"cArraySize\">0</hkparam>"
		"					<hkparam name=\"flags\">0</hkparam>"
		"					<hkparam name=\"offset\">36</hkparam>"
		"					<!-- attributes SERIALIZE_IGNORED -->"
		"				</hkobject>"
		"				<hkobject>"
		"					<hkparam name=\"name\">usedSimCloths</hkparam>"
		"					<hkparam name=\"class\">null</hkparam>"
		"					<hkparam name=\"enum\">null</hkparam>"
		"					<hkparam name=\"type\">TYPE_ARRAY</hkparam>"
		"					<hkparam name=\"subtype\">TYPE_UINT32</hkparam>"
		"					<hkparam name=\"cArraySize\">0</hkparam>"
		"					<hkparam name=\"flags\">0</hkparam>"
		"					<hkparam name=\"offset\">48</hkparam>"
		"					<!-- attributes SERIALIZE_IGNORED -->"
		"				</hkobject>"
		"			</hkparam>"
		"			<!-- defaults SERIALIZE_IGNORED -->"
		"			<!-- attributes SERIALIZE_IGNORED -->"
		"			<hkparam name=\"flags\">0</hkparam>"
		"			<hkparam name=\"describedVersion\">0</hkparam>"
		"		</hkobject>"
		""
		"		<hkobject name=\"#0012\" class=\"hkClass\" signature=\"0x14425e51\">"
		"			<hkparam name=\"name\">hclOperator</hkparam>"
		"			<hkparam name=\"parent\">#0003</hkparam>"
		"			<hkparam name=\"objectSize\">16</hkparam>"
		"			<hkparam name=\"numImplementedInterfaces\">0</hkparam>"
		"			<hkparam name=\"declaredEnums\" numelements=\"0\"></hkparam>"
		"			<hkparam name=\"declaredMembers\" numelements=\"2\">"
		"				<hkobject>"
		"					<hkparam name=\"name\">name</hkparam>"
		"					<hkparam name=\"class\">null</hkparam>"
		"					<hkparam name=\"enum\">null</hkparam>"
		"					<hkparam name=\"type\">TYPE_STRINGPTR</hkparam>"
		"					<hkparam name=\"subtype\">TYPE_VOID</hkparam>"
		"					<hkparam name=\"cArraySize\">0</hkparam>"
		"					<hkparam name=\"flags\">0</hkparam>"
		"					<hkparam name=\"offset\">8</hkparam>"
		"					<!-- attributes SERIALIZE_IGNORED -->"
		"				</hkobject>"
		"				<hkobject>"
		"					<hkparam name=\"name\">type</hkparam>"
		"					<hkparam name=\"class\">null</hkparam>"
		"					<hkparam name=\"enum\">null</hkparam>"
		"					<hkparam name=\"type\">TYPE_ENUM</hkparam>"
		"					<hkparam name=\"subtype\">TYPE_UINT32</hkparam>"
		"					<hkparam name=\"cArraySize\">0</hkparam>"
		"					<hkparam name=\"flags\">SERIALIZE_IGNORED</hkparam>"
		"					<hkparam name=\"offset\">12</hkparam>"
		"					<!-- attributes SERIALIZE_IGNORED -->"
		"				</hkobject>"
		"			</hkparam>"
		"			<!-- defaults SERIALIZE_IGNORED -->"
		"			<!-- attributes SERIALIZE_IGNORED -->"
		"			<hkparam name=\"flags\">0</hkparam>"
		"			<hkparam name=\"describedVersion\">0</hkparam>"
		"		</hkobject>"
		""
		"		<hkobject name=\"#0013\" class=\"hkClass\" signature=\"0x14425e51\">"
		"			<hkparam name=\"name\">hclTransformSetDefinition</hkparam>"
		"			<hkparam name=\"parent\">#0003</hkparam>"
		"			<hkparam name=\"objectSize\">20</hkparam>"
		"			<hkparam name=\"numImplementedInterfaces\">0</hkparam>"
		"			<hkparam name=\"declaredEnums\" numelements=\"0\"></hkparam>"
		"			<hkparam name=\"declaredMembers\" numelements=\"3\">"
		"				<hkobject>"
		"					<hkparam name=\"name\">name</hkparam>"
		"					<hkparam name=\"class\">null</hkparam>"
		"					<hkparam name=\"enum\">null</hkparam>"
		"					<hkparam name=\"type\">TYPE_STRINGPTR</hkparam>"
		"					<hkparam name=\"subtype\">TYPE_VOID</hkparam>"
		"					<hkparam name=\"cArraySize\">0</hkparam>"
		"					<hkparam name=\"flags\">0</hkparam>"
		"					<hkparam name=\"offset\">8</hkparam>"
		"					<!-- attributes SERIALIZE_IGNORED -->"
		"				</hkobject>"
		"				<hkobject>"
		"					<hkparam name=\"name\">type</hkparam>"
		"					<hkparam name=\"class\">null</hkparam>"
		"					<hkparam name=\"enum\">null</hkparam>"
		"					<hkparam name=\"type\">TYPE_INT32</hkparam>"
		"					<hkparam name=\"subtype\">TYPE_VOID</hkparam>"
		"					<hkparam name=\"cArraySize\">0</hkparam>"
		"					<hkparam name=\"flags\">0</hkparam>"
		"					<hkparam name=\"offset\">12</hkparam>"
		"					<!-- attributes SERIALIZE_IGNORED -->"
		"				</hkobject>"
		"				<hkobject>"
		"					<hkparam name=\"name\">numTransforms</hkparam>"
		"					<hkparam name=\"class\">null</hkparam>"
		"					<hkparam name=\"enum\">null</hkparam>"
		"					<hkparam name=\"type\">TYPE_UINT32</hkparam>"
		"					<hkparam name=\"subtype\">TYPE_VOID</hkparam>"
		"					<hkparam name=\"cArraySize\">0</hkparam>"
		"					<hkparam name=\"flags\">0</hkparam>"
		"					<hkparam name=\"offset\">16</hkparam>"
		"					<!-- attributes SERIALIZE_IGNORED -->"
		"				</hkobject>"
		"			</hkparam>"
		"			<!-- defaults SERIALIZE_IGNORED -->"
		"			<!-- attributes SERIALIZE_IGNORED -->"
		"			<hkparam name=\"flags\">0</hkparam>"
		"			<hkparam name=\"describedVersion\">0</hkparam>"
		"		</hkobject>"
		""
		"		<hkobject name=\"#0014\" class=\"hkClass\" signature=\"0x14425e51\">"
		"			<hkparam name=\"name\">hclBufferDefinition</hkparam>"
		"			<hkparam name=\"parent\">#0003</hkparam>"
		"			<hkparam name=\"objectSize\">28</hkparam>"
		"			<hkparam name=\"numImplementedInterfaces\">0</hkparam>"
		"			<hkparam name=\"declaredEnums\" numelements=\"0\"></hkparam>"
		"			<hkparam name=\"declaredMembers\" numelements=\"5\">"
		"				<hkobject>"
		"					<hkparam name=\"name\">name</hkparam>"
		"					<hkparam name=\"class\">null</hkparam>"
		"					<hkparam name=\"enum\">null</hkparam>"
		"					<hkparam name=\"type\">TYPE_STRINGPTR</hkparam>"
		"					<hkparam name=\"subtype\">TYPE_VOID</hkparam>"
		"					<hkparam name=\"cArraySize\">0</hkparam>"
		"					<hkparam name=\"flags\">0</hkparam>"
		"					<hkparam name=\"offset\">8</hkparam>"
		"					<!-- attributes SERIALIZE_IGNORED -->"
		"				</hkobject>";
	test_comp(xmldata, sizeof(xmldata));
	test_comp(xmldata, sizeof(xmldata), 200);
}

static void test_simple()
{
	static hkUchar data[BUFSIZE];

	{
		for (int i=0;i<1024 * 100; i++)
		{
			data[i] = (hkUchar)((i + 1) & 0xff);
		}
		test_comp(data, 1024 * 100);
	}

	{
		int i = 0;
		for (;i<1024 * 10; i++)
		{
			//runs of zeros
			data[i] = 0;
		}
		for (;i<1024 * 20; i++)
		{
			//repeated runs
			data[i] = (hkUchar)((i >> 12) & 0xff);
		}
		for (;i<1024 * 30; i++)
		{
			//random data
			data[i] = (hkUchar)((((i * 9483249) ^ 0x473285) + (i * 432523)) & 0xff);
		}
		test_comp(data, 1024 * 30);
		test_comp(data, 1024 * 30, 200);
	}
}

static void test_streams()
{
	hkArray<char>::Temp data(BUFSIZE);
	int test_buffer_sizes[] = {1<<16, 1<<10, 1<<8};
	for (int bufidx = 0; bufidx < (int)HK_COUNT_OF(test_buffer_sizes); bufidx ++)
	{	
		hkBufferedStreamWriter rawstream(data.begin(), data.getSize(), false);
		hkCompressedStreamWriter s(&rawstream, test_buffer_sizes[bufidx]);
		for (int i=0; i < 1000; i++)
		{
			s.write(&i, sizeof(i));
		}
		for (int i=0; i < 1000; i++)
		{
			int x = (i & 0x3f);
			s.write(&x, sizeof(x));
		}

		s.flush();

		hkMemoryStreamReader memread(data.begin(), data.getSize(), hkMemoryStreamReader::MEMORY_INPLACE);
		hkCompressedStreamReader r(&memread);
		for (int i=0; i < 1000; i++)
		{
			int i2;
			r.read(&i2, sizeof(i2));
			HK_TEST(i == i2);
		}
		for (int i=0; i < 1000; i++)
		{
			int i2;
			r.read(&i2, sizeof(i2));
			HK_TEST((i & 0x3f) == i2);
		}
	}
}

int compression_main()
{
	test_streams();
	test_xmldata();
	test_simple();
	test_dataset();

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(compression_main, "Slow", "Common/Test/UnitTest/Base/", __FILE__);

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
