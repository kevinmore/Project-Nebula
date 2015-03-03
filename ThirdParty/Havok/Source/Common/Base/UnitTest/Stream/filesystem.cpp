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
#include <Common/Base/System/Io/Reader/hkStreamReader.h>
//#include <stdio.h>

static void fs_iterator()
{
	hkFileSystem& fs = hkFileSystem::getInstance();

	hkFileSystem::Iterator iter(&fs, "Resources/Common/Textures");
	int count = 0;
	bool recursed = false;
	while( iter.advance() )
	{
		count += 1;

		const hkFileSystem::Entry& e = iter.current();
		//printf("%10x, %s\n", int(e.getMtime()>>32), e.getPath());
		if(e.isDir() && !recursed)
		{
			iter.recurseInto( e.getPath() );
		}
	}
	HK_TEST( count != 0 );
}

static void fs_openwrite()
{
	hkFileSystem& fs = hkFileSystem::getInstance();
	{
		hkRefPtr<hkStreamWriter> w0 = fs.openWriter("test.txt");
		w0->write("helloblah",9);
	}
	{
		hkRefPtr<hkStreamReader> r0 = fs.openReader("test.txt");
		char buf[100] = {};
		int n = r0->read(buf, sizeof(buf)-1);
		HK_TEST(n == 9);
		HK_TEST( hkString::strCmp(buf, "helloblah") == 0);
	}
}

static void fs_append()
{
	hkFileSystem& fs = hkFileSystem::getInstance();
	{
		hkRefPtr<hkStreamWriter> w0 = fs.openWriter("test.txt");
		w0->write("hello",5);
	}
	{
		hkRefPtr<hkStreamWriter> w0 = fs.openWriter("test.txt", hkFileSystem::OpenFlags(hkFileSystem::OPEN_DEFAULT_WRITE & (~hkFileSystem::OPEN_TRUNCATE)) );
		w0->seek(0, hkStreamWriter::STREAM_END);
		w0->write("world",5);
	}
	{
		hkRefPtr<hkStreamReader> r0 = fs.openReader("test.txt");
		char buf[100] = {};
		int n = r0->read(buf, sizeof(buf)-1);
		HK_TEST(n == 10);
		HK_TEST( hkString::strCmp(buf, "helloworld") == 0);
	}
}

int filesystem_main()
{
	fs_openwrite();

	// 2013.2 is getting lots of testing spam from some platforms because the filesystem isn't fully implemented
	// disable them for now. Do not merge this back to the HEAD
#if defined(HK_PLATFORM_WIN32) || defined(HK_PLATFORM_DURANGO) || defined(HK_PLATFORM_PS3) || defined(HK_PLATFORM_PS4) || defined(HK_PLATFORM_XBOX360) || defined(HK_PLATFORM_MAC386) || defined(HK_PLATFORM_LINUX)
	fs_iterator();
	fs_append();
#endif
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(filesystem_main, "Fast", "Common/Test/UnitTest/Base/", __FILE__     );

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
