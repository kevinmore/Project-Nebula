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

static bool fileFound(hkFileSystem& fs, const char* filename, const char* dirname, const char* filter)
{
	hkBool found = false;

	hkFileSystem::Iterator dirIter(&fs, dirname, filter);
	while(dirIter.advance())
	{
		const hkFileSystem::Entry entry(dirIter.current());
		if(entry.isFile())
		{

			if(!hkString::strCmp(entry.getName(), filename))
			{
				HK_TEST(!found); found = true;
			}
		}
	}
	return found;
}

int fileops_main()
{
	// 2013.2 is getting lots of testing spam from some platforms because the filesystem isn't fully implemented
	// disable them for now. Do not merge this back to the HEAD
#if defined(HK_PLATFORM_WIN32) || defined(HK_PLATFORM_DURANGO) || defined(HK_PLATFORM_PS3) || defined(HK_PLATFORM_PS4) || defined(HK_PLATFORM_XBOX360) || defined(HK_PLATFORM_MAC386) || defined(HK_PLATFORM_LINUX)
	hkFileSystem& fs = hkFileSystem::getInstance();
	hkStringBuf originalBuffer = "Hello File System\n\nFile System\n";

	{
		{
			hkRefPtr<hkStreamWriter> sw = fs.openWriter("testFileSystem.txt", hkFileSystem::OPEN_DEFAULT_WRITE);
			HK_TEST(sw->write(originalBuffer.cString(), originalBuffer.getLength()) == originalBuffer.getLength());
		}
		{
			hkRefPtr<hkStreamWriter> sw = fs.openWriter("testFileSystem2.txt", hkFileSystem::OPEN_DEFAULT_WRITE);
			HK_TEST(sw->write(originalBuffer.cString(), originalBuffer.getLength()) == originalBuffer.getLength());
		}
		{
			hkRefPtr<hkStreamWriter> sw = fs.openWriter("testFileSystem3.txt", hkFileSystem::OPEN_DEFAULT_WRITE);
			HK_TEST(sw->write(originalBuffer.cString(), originalBuffer.getLength()) == originalBuffer.getLength());
		}
	}

	// Test list directory
	{
		HK_TEST(fileFound(fs, "testFileSystem.txt", "", "*.txt"));
		HK_TEST(fileFound(fs, "testFileSystem2.txt", "", HK_NULL));
		HK_TEST(!fileFound(fs, "testFileSystem2.txt", "", "*.doesntexist"));
		HK_TEST(fileFound(fs, "testFileSystem3.txt", "", "*.txt"));
	}

	// Test simple read
	{
		hkRefPtr<hkStreamReader> sr = fs.openReader("testFileSystem.txt", hkFileSystem::OPEN_DEFAULT_READ);

		char readBuffer[1024];
		HK_TEST(sr->read(readBuffer, 17) == 17);
		readBuffer[17] = 0;
		HK_TEST(!hkString::strCmp(readBuffer, "Hello File System"));
		HK_TEST(sr->read(readBuffer, 1024) == 14);
		readBuffer[14] = 0;
		HK_TEST(!hkString::strCmp(readBuffer, "\n\nFile System\n"));
	}
	HK_TEST(fs.remove("testFileSystem.txt") == hkFileSystem::RESULT_OK);

	// Test list directory after file removed
	{
		HK_TEST(!fileFound(fs, "testFileSystem.txt", "", HK_NULL));
		HK_TEST(fileFound(fs, "testFileSystem2.txt", "", "*.txt"));
		HK_TEST(fileFound(fs, "testFileSystem3.txt", "", "*.txt"));
	}

	// Test buffered read
	{
		hkRefPtr<hkStreamReader> containingSr = fs.openReader("testFileSystem2.txt", hkFileSystem::OPEN_DEFAULT_READ);
		hkSeekableStreamReader* sr = containingSr->isSeekTellSupported();
		HK_TEST(sr != HK_NULL);
		char readBuffer[1024];
		HK_TEST(sr->peek(readBuffer, 1) == 1); // Peek 'H'
		HK_TEST(readBuffer[0] == 'H');
		HK_TEST(sr->read(readBuffer, 6) == 6); // Read over "Hello "
		readBuffer[6] = 0;
		HK_TEST(!hkString::strCmp(readBuffer, "Hello "));
		HK_TEST(sr->peek(readBuffer, 4) == 4); // Peek "File"
		readBuffer[4] = 0;
		HK_TEST(!hkString::strCmp(readBuffer, "File"));
		// Seek back to start
		HK_TEST(sr->seek(0, hkSeekableStreamReader::STREAM_SET) == HK_SUCCESS);
		HK_TEST(sr->peek(readBuffer, 5) == 5); // Peek "Hello"
		readBuffer[5] = 0;
		HK_TEST(!hkString::strCmp(readBuffer, "Hello"));
		HK_TEST(sr->read(readBuffer, 17) == 17); // Read over "Hello File System"
		readBuffer[17] = 0;
		HK_TEST(!hkString::strCmp(readBuffer, "Hello File System"));
		HK_TEST(sr->peek(readBuffer, 1024) == 14); // Peek to end of file
		readBuffer[14] = 0;
		HK_TEST(!hkString::strCmp(readBuffer, "\n\nFile System\n"));

		HK_TEST(sr->isOk()); // Peek has not finished file

		HK_TEST(sr->read(readBuffer, 1024) == 14); // Read to end of file
		readBuffer[14] = 0;
		HK_TEST(!hkString::strCmp(readBuffer, "\n\nFile System\n"));

		HK_TEST(sr->seek(-7, hkSeekableStreamReader::STREAM_END) == HK_SUCCESS); // Seek back a few bytes
		HK_TEST(sr->read(readBuffer, 1024) == 7); // Read to end of file
		readBuffer[7] = 0;
		HK_TEST(!hkString::strCmp(readBuffer, "System\n"));

		HK_TEST(sr->peek(readBuffer, 1024) == 0); // Nothing left to peek
		HK_TEST(!sr->isOk()); // Peek finishes the file
		
		HK_TEST(sr->read(readBuffer, 1024) == 0); // Nothing left to read
		HK_TEST(!sr->isOk()); // File finished
	}

	HK_TEST(fs.remove("testFileSystem2.txt") == hkFileSystem::RESULT_OK);
	
	// Test open append
	{
		hkStreamWriter* sw = fs.openWriter("testFileSystem3.txt", hkFileSystem::OpenFlags(hkFileSystem::OPEN_DEFAULT_WRITE & (~hkFileSystem::OPEN_TRUNCATE))).stealOwnership();
		sw->seek(0, hkStreamWriter::STREAM_END);
		hkStringBuf appendText("Appended to file\n");
		sw->write(appendText.cString(), appendText.getLength());
		sw->removeReference();

		char readBuffer[1024];

		hkRefPtr<hkStreamReader> sr = fs.openReader("testFileSystem3.txt", hkFileSystem::OPEN_DEFAULT_READ);
		HK_TEST(sr->peek(readBuffer, 1024) == 48); // Peek all of file
		readBuffer[48] = 0;
		HK_TEST(!hkString::strCmp(readBuffer, "Hello File System\n\nFile System\nAppended to file\n"));

		HK_TEST(sr->read(readBuffer, 1024) == 48); // Read to end of file
		readBuffer[48] = 0;
		HK_TEST(!hkString::strCmp(readBuffer, "Hello File System\n\nFile System\nAppended to file\n"));

	}
	
	HK_TEST(fs.remove("testFileSystem3.txt") == hkFileSystem::RESULT_OK);

	// Test list directory after file removed
	{
		HK_TEST(!fileFound(fs, "testFileSystem.txt", "", HK_NULL));
		HK_TEST(!fileFound(fs, "testFileSystem2.txt", "", HK_NULL));
		HK_TEST(!fileFound(fs, "testFileSystem3.txt", "", HK_NULL));
	}

	// Test directory iterator on known files
	{
		hkFileSystem::Iterator dirIter(&fs, "Resources/Common/FileSystem");
		hkArray<hkFileSystem::Entry> files;
		while(dirIter.advance())
		{
			hkFileSystem::Entry entry(dirIter.current());
			if(entry.isFile())
			{
				files.pushBack(entry);
			}
			else if(entry.isDir())
			{
				dirIter.recurseInto(entry.getPath());
			}
		}
		
		// This dir should contain 3 files, of size 11. Each file contents is equal to its name
		HK_TEST(files.getSize() == 3);
		while(files.getSize())
		{
			hkFileSystem::Entry file = files.back();
			hkRefPtr<hkStreamReader> sr = fs.openReader(file.getPath(), hkFileSystem::OPEN_DEFAULT_READ);
			HK_TEST(file.getSize() == 11);
			HK_TEST(file.isFile());
			char readBuffer[1024];
			HK_TEST(sr->read(readBuffer, 1024) == 11); // Peek all of file
			readBuffer[11] = 0;
			HK_TEST(!hkString::strCmp(readBuffer, file.getName()));
			files.popBack();
		}
	}

	// Test seek past the end of the file fills with zero
	{
		hkStringBuf shortString("In File");
		{
			hkRefPtr<hkStreamWriter> sw = fs.openWriter("testSeekPastEnd.txt", hkFileSystem::OPEN_DEFAULT_WRITE);
			HK_TEST(sw->write(shortString.cString(), shortString.getLength()) == shortString.getLength());

			// Seek past the end of the file should fill with zeros
			HK_TEST(sw->seek(50, hkStreamWriter::STREAM_SET) == HK_SUCCESS);
			HK_TEST(sw->write(shortString.cString(), shortString.getLength()) == shortString.getLength());
		}

		char readBuffer[1024];
		// First string
		{
			hkRefPtr<hkStreamReader> sr = fs.openReader("testSeekPastEnd.txt", hkFileSystem::OPEN_DEFAULT_READ);

			HK_TEST(sr->read(readBuffer, 7) == 7);
			readBuffer[7] = 0;
			HK_TEST(!hkString::strCmp(readBuffer, "In File"));

			HK_TEST(sr->read(readBuffer, 43) == 43);
			
			// Xbox 360 does not initialize the intervening bytes
#if !defined(HK_PLATFORM_XBOX360)
			// The filled in bytes should all be zero
			for(int i=0;i<43;i++)
			{
				HK_TEST(readBuffer[i] == 0);
			}
#endif

			// Then the write occurs after the seek
			HK_TEST(sr->read(readBuffer, 1024) == 7);
			readBuffer[7] = 0;
			HK_TEST(!hkString::strCmp(readBuffer, "In File"));
		}

		HK_TEST(fs.remove("testSeekPastEnd.txt") == hkFileSystem::RESULT_OK);
	}
#endif
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif

HK_TEST_REGISTER(fileops_main,     "Fast", "Common/Test/UnitTest/Base/",     __FILE__    );

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
