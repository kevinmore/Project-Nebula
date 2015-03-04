/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#include <Common/Base/hkBase.h>
#include <Common/Base/Types/hkBaseTypes.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>
#include <Common/Base/DebugUtil/DeterminismUtil/hkCheckDeterminismUtil.h>
#include <Common/Base/System/Io/IStream/hkIStream.h>
#include <Common/Base/System/Io/Writer/Array/hkArrayStreamWriter.h>
#include <Common/Base/System/Io/Writer/Crc/hkCrcStreamWriter.h>
#include <Common/Base/System/Io/FileSystem/hkFileSystem.h>
#include <Common/Base/Thread/CriticalSection/hkCriticalSection.h>
#include <Common/Base/Memory/System/hkMemorySystem.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Base/Thread/JobQueue/hkJobQueue.h>

#ifdef HK_PLATFORM_WIN32
	#include "stdio.h"
#endif

#ifdef LODEPNG_H
	#include <vector>
#endif

extern HK_THREAD_LOCAL( int ) hkThreadNumber;
hkCheckDeterminismUtil* g_checkDeterminismUtil;

int hkCheckDeterminismUtil_id;
hkReal* hkCheckDeterminismUtil_reference;
hkReal* hkCheckDeterminismUtil_object;
int hkCheckDeterminismUtil_size;
hkReal* hkCheckDeterminismUtil_crcObject;

hkUint32 hkCheckDeterminismUtil::s_heapAllocScrubValueWrite = 0x7ffa110c; // ALLOC
hkUint32 hkCheckDeterminismUtil::s_heapFreeScrubValueWrite = 0x7fffefef; // FREE
hkUint32 hkCheckDeterminismUtil::s_heapAllocScrubValueCheck = 0x0; // ALLOC
hkUint32 hkCheckDeterminismUtil::s_heapFreeScrubValueCheck = 0x0; // FREE
hkUint32 hkCheckDeterminismUtil::s_stackScrubValueWrite = 0x7ffdadad;
hkUint32 hkCheckDeterminismUtil::s_stackScrubValueCheck = 0x0;

// put these into a structure
static HK_THREAD_LOCAL(hkIstream*) m_inputStream;
static HK_THREAD_LOCAL(hkOstream*) m_outputStream;
static HK_THREAD_LOCAL(void*)     m_jobInfoIdx;
static HK_THREAD_LOCAL(void*)     m_isPrimaryWorkerThread;

static HK_THREAD_LOCAL(hkUint32)     m_currentJobFuid0;
static HK_THREAD_LOCAL(hkUint32)     m_currentJobFuid1;
static HK_THREAD_LOCAL(hkUint32)     m_currentJobFuid2;
static HK_THREAD_LOCAL(hkUint32)     m_currentJobFuid3;
static HK_THREAD_LOCAL(hkUint32)     m_currentCheckIndex;

bool hkCheckDeterminismUtil::Fuid::operator==(const hkCheckDeterminismUtil::Fuid& f)
{
	return m_0 == f.m_0 && m_jobPackedId == f.m_jobPackedId && m_2 == f.m_2 && m_3 == f.m_3 && m_4 == f.m_4;
}

bool hkCheckDeterminismUtil::Fuid::operator!=(const hkCheckDeterminismUtil::Fuid& f)
{
	return m_0 != f.m_0 || m_jobPackedId != f.m_jobPackedId || m_2 != f.m_2 || m_3 != f.m_3 || m_4 != f.m_4;
}

void hkCheckDeterminismUtil::Fuid::setPackedJobId(const struct hkJob& job)
{
	m_jobPackedId = ((hkUint16(job.m_jobType) << 8) | hkUint16(job.m_jobSubType));
}

hkCheckDeterminismUtil::hkCheckDeterminismUtil()
{
	m_inSingleThreadedCode = true;
	m_shared = new hkCriticalSection();

	m_sharedInputStream  = HK_NULL;
	m_sharedOutputStream = HK_NULL;
	HK_THREAD_LOCAL_SET(m_inputStream, m_sharedInputStream);
	HK_THREAD_LOCAL_SET(m_outputStream, m_sharedOutputStream);
	m_primaryWorkerThreadInputStream = HK_NULL;
	m_primaryWorkerThreadOutputStream = HK_NULL;
	m_memoryTrack = HK_NULL;

	m_frame = 0;

	// Debug flag
	m_writingStFromWorker = false;
	m_mode = MODE_IDLE;


#ifdef HK_PLATFORM_WIN32
	// Disable output buffering for determinism error reports.
	setbuf(stdout, NULL);
#endif

	// Job delay
	m_delayJobs = false;
	m_delayCounter = 1024*256;
	if (m_delayJobs)
	{
		#ifdef LODEPNG_H
		{
			std::vector<unsigned char> image; 
			unsigned width, height;
			lodepng::decode(image, width, height, "pattern.png");

			m_delayJobSeed.setSize(image.size());
			for (unsigned int i=0;i<image.size(); ++i)
			{
				m_delayJobSeed[i] = image[i];
			}
		}
		#endif

		m_delayJobs = !m_delayJobSeed.isEmpty();
	}

	// Thread tracking
	m_enableThreadTracker = false;
	m_threadTracker.setSize(32);
	m_maxTheadId = 0;
}


hkCheckDeterminismUtil::~hkCheckDeterminismUtil()
{
	HK_ASSERT2(0xad876dda, m_sharedInputStream == HK_THREAD_LOCAL_GET(m_inputStream), "Upon destruction, the thread-local streams are expected to be set to the shared streams (i.e. working in single-threaded mode)." );
	HK_ASSERT2(0xad876dda, m_sharedOutputStream == HK_THREAD_LOCAL_GET(m_outputStream), "Upon destruction, the thread-local streams are expected to be set to the shared streams (i.e. working in single-threaded mode)." );

	finish();

	delete m_shared;

	// Thread local structures
	HK_THREAD_LOCAL_SET(m_inputStream, HK_NULL);
	HK_THREAD_LOCAL_SET(m_outputStream, HK_NULL);
	HK_THREAD_LOCAL_SET(m_jobInfoIdx, HK_NULL);
	HK_THREAD_LOCAL_SET(m_isPrimaryWorkerThread, HK_NULL);
	HK_THREAD_LOCAL_SET(m_currentJobFuid0, HK_NULL);
	HK_THREAD_LOCAL_SET(m_currentJobFuid1, HK_NULL);
	HK_THREAD_LOCAL_SET(m_currentJobFuid2, HK_NULL);
	HK_THREAD_LOCAL_SET(m_currentJobFuid3, HK_NULL);
}

hkCheckDeterminismUtil::Fuid& hkCheckDeterminismUtil::Fuid::getZeroFuid()
{
	static Fuid fuid;
	fuid.m_0 = hkUint32(-1);
	fuid.m_jobPackedId = hkUint16(-1);
	fuid.m_2 = 0;
	fuid.m_3 = 0;
	
	return fuid;
}

hkCheckDeterminismUtil::Fuid& hkCheckDeterminismUtil::Fuid::getCanceledFuid()
{
	static Fuid fuid;
	fuid.m_0 = hkUint32(-2);
	fuid.m_jobPackedId = hkUint16(-1);
	fuid.m_2 = 0;
	fuid.m_3 = 0;
	
	return fuid;
}

extern "C" hkSystemTime HK_CALL hkGetSystemTime();
namespace
{
	static void makeFilename(const char* filename, bool stampFilename, hkStringPtr& ptrOut)
	{
		hkStringBuf buf;
#if defined(HK_PLATFORM_WIN32) && !defined(HK_PLATFORM_WINRT)
		if (stampFilename)
		{
			char szFileName[MAX_PATH];
			GetModuleFileNameA( NULL, szFileName, MAX_PATH );
			buf.set(szFileName);
			buf.pathNormalize();
			buf.pathBasename();
			buf.appendPrintf("_%d_%d.bin", hkInt32(GetCurrentProcessId()), hkUint32(hkGetSystemTime()));
		}
		else
#endif
		{
			buf.printf("%s", filename);
		}

		ptrOut = buf.cString();
	}
}

void hkCheckDeterminismUtil::startWriteMode(bool stampFilename, const char* filename)
{
#if defined (HK_ENABLE_DETERMINISM_CHECKS)
	HK_ASSERT2(0xaf36affe, !m_sharedInputStream, "You cannot switch to WRITE mode without calling finish() first.");
	
	// set scrub values for write
	hkMemorySystem& memorySystem = hkMemorySystem::getInstance();
	memorySystem.setHeapScrubValues(s_heapAllocScrubValueWrite, s_heapFreeScrubValueWrite);

	if ( !filename )
	{
		HK_ASSERT( 0xf0ed3dfe, !m_memoryTrack );
		m_memoryTrack = new hkMemoryTrack;
		m_sharedOutputStream = new hkOstream(m_memoryTrack);
	}
	else
	{
		makeFilename(filename, stampFilename, m_filename);
		HK_REPORT(m_filename.cString());
		m_sharedOutputStream = new hkOstream(m_filename.cString());
	}

	HK_ASSERT2(0xaf36affd, m_sharedOutputStream->isOk(), "Output file could not be opened.");

	m_mode = MODE_WRITE;

	HK_THREAD_LOCAL_SET(m_inputStream, m_sharedInputStream);
	HK_THREAD_LOCAL_SET(m_outputStream, m_sharedOutputStream);

	m_frame = 0;
#endif
}


void hkCheckDeterminismUtil::startCheckMode(const char* filename)
{
#if defined (HK_ENABLE_DETERMINISM_CHECKS)
	HK_ASSERT2(0xaf36affe, !m_sharedOutputStream, "You cannot switch to READ mode without calling finish() first.");

	// set scrub values for check
	hkMemorySystem& memorySystem = hkMemorySystem::getInstance();
	memorySystem.setHeapScrubValues(s_heapAllocScrubValueCheck, s_heapFreeScrubValueCheck);

	if ( m_memoryTrack )
	{
		m_sharedInputStream = new hkIstream(m_memoryTrack);
	}
	else
	{
		m_sharedInputStream = new hkIstream(filename ? filename : m_filename.cString());
	}


	if ( !m_sharedInputStream->isOk() )
	{
		HK_ASSERT2(0xaf36affe, false, "Input file not found.");
		finish();
	}
	else
	{
		m_mode = MODE_COMPARE;
	}

	HK_THREAD_LOCAL_SET(m_inputStream,  m_sharedInputStream);
	HK_THREAD_LOCAL_SET(m_outputStream, m_sharedOutputStream);

	m_frame = 0;
#endif
}

struct IntPair
{
	int m_id;
	int m_size;
	inline hkBool operator<(const IntPair& other) const { return m_size < other.m_size; }
};

void hkCheckDeterminismUtil::finish()
{
#if defined (HK_ENABLE_DETERMINISM_CHECKS)
	if ( m_sharedInputStream )
	{
		// check whether we reached the end of the file
		{
			char tmpBuffer[4];	m_sharedInputStream->read(tmpBuffer, 1);
			HK_ASSERT(0xad87b754, !m_sharedInputStream->isOk());
		}
		delete m_sharedInputStream;
		m_sharedInputStream = HK_NULL;
		delete m_memoryTrack;
		m_memoryTrack = HK_NULL;
	}
	else if ( m_sharedOutputStream )
	{
		m_sharedOutputStream->flush();
		delete m_sharedOutputStream;
		m_sharedOutputStream = HK_NULL;
		delete m_memoryTrack;
		m_memoryTrack = HK_NULL;
#if defined(HK_DETERMINISM_CHECK_SIZES)
		{
			hkArray<IntPair> pairs;
			for ( hkPointerMap<int,int>::Iterator i = m_sizePerId.getIterator(); m_sizePerId.isValid(i); i = m_sizePerId.getNext(i ))
			{
				IntPair& p = pairs.expandOne();
				p.m_id = m_sizePerId.getKey( i );
				p.m_size = m_sizePerId.getValue( i );
			}
			hkSort( pairs.begin(), pairs.getSize() );
			for (int k = pairs.getSize()-1; k>=0; k--)
			{
				IntPair& p = pairs[k];
				char buffer[256];
				hkString::sprintf( buffer, "hkCheckDeterminismUtil::id %x uses %i bytes", p.m_id, p.m_size );
				HK_REPORT( buffer );
			}

		}
#endif
	}

	HK_THREAD_LOCAL_SET(m_inputStream,  HK_NULL);
	HK_THREAD_LOCAL_SET(m_outputStream, HK_NULL);

#ifdef HK_PLATFORM_WIN32
	if (m_mode == MODE_COMPARE)
	{
		hkFileSystem::getInstance().remove(m_filename.cString());
	}
#endif

	m_mode = MODE_IDLE;

#endif 
}

void hkCheckDeterminismUtil::flushWrite()
{
#if defined (HK_ENABLE_DETERMINISM_CHECKS)
	if ( m_sharedOutputStream )
	{
		m_sharedOutputStream->flush();
	}

#endif 
}

namespace 
{

static void initializeBreakpoint(struct Breakpoint& b);

struct Breakpoint
{
	hkUint64						frame;
	hkCheckDeterminismUtil::Fuid	fuid;
	hkUint64						checkIndex;	
	int								size;
	int								dataErrorIndex;
	hkUint8							d[1024];
	hkUint8							od[1024];

	Breakpoint() {}
	
	static const Breakpoint& get()
	{
		static Breakpoint b;
		static bool initialized = false;

		if (!initialized)
		{
			hkString::memSet(&b, 0, sizeof(b));
			b.fuid = hkCheckDeterminismUtil::Fuid::getZeroFuid();
			
			initializeBreakpoint(b);
		}

		return b;
	}

	static void print(hkStringBuf& str, hkUint64 frame, const char* sourceObject, const char* oldObject, int size, int index, hkCheckDeterminismUtil::Fuid* threadFuids, int maxThreadId)
	{
		hkCheckDeterminismUtil::Fuid fuid = hkCheckDeterminismUtil::getCurrentJobFuid();
		hkUint64 checkIndex = hkCheckDeterminismUtil::getCurrentCheckIndex();

		str.appendPrintf("#if 1\n{\n");
		str.appendPrintf("\n\n////// Determinism Breakpoint START //////\n");
		str.appendPrintf("// place this code into hkCheckDeterminismUtil.cpp, initializeBreakpoint()\n");
		str.appendPrintf("#define HK_DETERMINISM_ENABLE_BREAKPOINT\n");
		str.appendPrintf("b.frame = %u;\n", frame);
		str.appendPrintf("b.fuid.m_0= %u; b.fuid.m_jobPackedId= %u; b.fuid.m_2= %u; b.fuid.m_3= %u; b.fuid.m_4= %u; // <%u, %u, %u, %u, %u>\n", 
							(hkUint32) fuid.m_0, (hkUint32) fuid.getPackedJobId(), (hkUint32) fuid.m_2, (hkUint32) fuid.m_3, (hkUint32) fuid.m_4,
							(hkUint32) fuid.m_0, (hkUint32) fuid.getPackedJobId(), (hkUint32) fuid.m_2, (hkUint32) fuid.m_3, (hkUint32) fuid.m_4);
		str.appendPrintf("b.checkIndex = %u;\n", hkUint32(checkIndex));
		str.appendPrintf("b.size = %d;\n", size);
		str.appendPrintf("b.dataErrorIndex = %d;\n", index);

		str.appendPrintf("// New data\n");
		for (int i=0; i<size; ++i)
		{
			str.appendPrintf("b.d[%d]=%d;", i, hkUint8(sourceObject[i]));
			if ((i > 0) && ((i % 32) == 0)) str.appendPrintf("\n");
		}
		str.appendPrintf("\n");

		str.appendPrintf("// Old data\n");
		for (int i=0; i<size; ++i)
		{
			str.appendPrintf("b.od[%d]=%d;", i, hkUint8(oldObject[i]));
			if ((i > 0) && ((i % 32) == 0)) str.appendPrintf("\n");
		}
		str.appendPrintf("\n");
		
		if (maxThreadId > 0)
		{
			hkCheckDeterminismUtil::Fuid invalidFuid = hkCheckDeterminismUtil::Fuid::getZeroFuid();
			invalidFuid.m_0 = 0;

			int tid = HK_THREAD_LOCAL_GET( hkThreadNumber );
			str.appendPrintf("/* Thread Fuids\n");
			for (int i=0; i<=maxThreadId; ++i)
			{
				if (threadFuids[i] != invalidFuid)
				{
					str.appendPrintf("[ %d%s] <%u, %u, %u, %u, %u>\n", i, tid == i ? "*": " ",
							(hkUint32) threadFuids[i].m_0, (hkUint32) threadFuids[i].getPackedJobId(), (hkUint32) threadFuids[i].m_2, (hkUint32) threadFuids[i].m_3, (hkUint32) threadFuids[i].m_4);
				}
			}
			str.appendPrintf("*/\n");
		}

		str.appendPrintf("////// Determinism Breakpoint END //////\n");
		str.appendPrintf("#endif\n\n");
	}
};

// When there is a set breakpoint, its generated could should go into this function. Otherwise, it should be totally empty.
static void initializeBreakpoint(struct Breakpoint& b)
{
}


static void checkBreakpoint(const hkUint64& frame, const void* object)
{
#ifdef HK_DETERMINISM_ENABLE_BREAKPOINT
	const Breakpoint& bkpt = Breakpoint::get();
			
	hkCheckDeterminismUtil::Fuid fuid = hkCheckDeterminismUtil::getCurrentJobFuid();
	hkUint32 checkIndex = hkCheckDeterminismUtil::getCurrentCheckIndex();

	//printf("frame:%d ", frame);
	//printf("fuid:%u, %u, %u, %u, %u ", hkUint32(fuid.m_0), hkUint32(fuid.m_1), hkUint32(fuid.m_2), hkUint32(fuid.m_3), hkUint32(fuid.m_4));
	//printf("ci:%u\n", hkUint32(checkIndex));

	if (frame == bkpt.frame)
	{
		if (fuid == bkpt.fuid)
		{
			if (checkIndex == bkpt.checkIndex)
			{
				// some variables helping debugging
				const hkReal* of = (const hkReal*)object; (void)of;
				const hkReal* cf = (const hkReal*)bkpt.d; (void)cf;
				const void*const* oh = (const void*const*)of; (void)oh;
				const void*const* ch = (const void*const*)cf; (void)ch;

				hkStringBuf text;
				text.printf("\nDeterminism breakpoint reached");
				HK_REPORT(text.cString());	
			}
			else
			{
				if (checkIndex+1 == bkpt.checkIndex)
				{
					hkStringBuf text;
					text.printf("\nDeterminism pre-breakpoint reached");
					HK_REPORT(text.cString());	
				}
			}
		}
	}
#endif
}


} // anonymous namespace


bool hkCheckDeterminismUtil::isNearBreakpoint(hkUint64 offset)
{
#ifdef HK_DETERMINISM_ENABLE_BREAKPOINT
	const Breakpoint& bkpt = Breakpoint::get();
	hkCheckDeterminismUtil::Fuid fuid = getCurrentJobFuid();
	hkUint32 checkIndex = hkCheckDeterminismUtil::getCurrentCheckIndex();

	if (m_frame == bkpt.frame)
	{
		if (fuid == bkpt.fuid)
		{
			if (checkIndex <= bkpt.checkIndex && checkIndex + offset >= bkpt.checkIndex)
			{	
				return true;
			}
		}
	}
	
	return false;
#else
	return false;
#endif
}


void hkCheckDeterminismUtil::checkImpl(int id, const void* object, int size, int* excluded)
{
	if (!size)
	{
		return;
	}
	const char* sourceObject = (const char*)object;

	hkOstream* outputStream = HK_THREAD_LOCAL_GET(m_outputStream);


	if ( outputStream )
	{
#if defined(HK_DETERMINISM_CHECK_SIZES)
		int currentSize = m_sizePerId.getWithDefault( id, 0 );
		currentSize += size;
		m_sizePerId.insert( id, currentSize );
#endif
		outputStream->write(sourceObject, size);
		HK_ASSERT( 0xf0323446, outputStream->isOk() );
		
		checkBreakpoint(m_frame, object);

		bumpCurrentCheckIndex();
		return;
	}

	// Setup the exclusion variable.
	int nextExludedOffsetIndex = ( excluded == HK_NULL ? -1 : 0 );
	int nextExcludedOffset = ( excluded == HK_NULL ? -1 : excluded[0] );

	hkIstream* inputStream = HK_THREAD_LOCAL_GET(m_inputStream);
	if (inputStream == HK_NULL)
	{
		HK_ASSERT2(0xad7655dd, false, "Neither stream exists.");
		bumpCurrentCheckIndex();
		return;
	}

	hkLocalBuffer<char> readBuffer(size);
	HK_ON_DEBUG( int debug_numRead = )inputStream->read(readBuffer.begin(), size);
	// Note that we don't check ::isOk() here, since we might be at the end of the file.
	HK_ASSERT( 0xf0323445, debug_numRead == size);

	// some variables helping debugging
	const hkReal* of = (const hkReal*)object; (void)of;
	const hkReal* cf = (const hkReal*)readBuffer.begin(); (void)cf;
	const void*const* oh = (const void*const*)of; (void)oh;
	const void*const* ch = (const void*const*)cf; (void)ch;

	for (int i=0; i<size; i++)
	{
		// Compare the bytes.
		if (i != nextExcludedOffset)
		{	
			if((sourceObject[i] != readBuffer[i]))
			{								
				hkCheckDeterminismUtil_id = id;
				hkCheckDeterminismUtil_object = (hkReal*)object;
				hkCheckDeterminismUtil_reference = const_cast<hkReal*>(cf);
				hkCheckDeterminismUtil_size = size;

				// Look ahead in the input stream. Can be useful if you use a string as a marker.
				hkLocalArray<char> futureBuffer(1024);
				int numRead = inputStream->read( futureBuffer.begin(), 1024);
				futureBuffer.setSize(numRead);
			
				hkStringBuf bkptStr;
				Breakpoint::print(bkptStr, m_frame, sourceObject, readBuffer.begin(), size, i, m_threadTracker.begin(), m_maxTheadId);
				hkError::messageReport(-1, bkptStr.cString(), HK_CURRENT_FILE, __LINE__);
				
				hkStringBuf text;
				text.printf("\nDeterminism check failed: size %d, i %d, obj 0x%08X, ref 0x%08X, obj[i] 0x%02X, ref[i] 0x%02X", size, i, 
							(void*) sourceObject, (void*) readBuffer.begin(), sourceObject[i], readBuffer[i]);
				//printf(text.cString());
				HK_ERROR(id, text.cString());			
				//HK_REPORT(text.cString());			
			}
		}
		else
		{
			// Skip the required number of excluded bytes.
			i += excluded[nextExludedOffsetIndex+1] - 1;
			// Advance to the next exclusion.
			nextExludedOffsetIndex += 2;
			// Set the next excluded offset.
			nextExcludedOffset = excluded[nextExludedOffsetIndex];
			// Make sure the excluded array is sorted by offset.
			HK_ASSERT2(0xef6e13a6, (nextExcludedOffset == -1) || (nextExcludedOffset >= i), "the 'excluded' array has to be sorted by offset.");
		}
	}
		  
	bumpCurrentCheckIndex();
	return;
}

void hkCheckDeterminismUtil::checkCrcImpl( int id, const void* object, int size)
{
	hkCrc32StreamWriter crcWriter;
	crcWriter.write( object, size );
	int crc = crcWriter.getCrc();
	hkCheckDeterminismUtil_crcObject = (hkReal*)object;
	checkMt( id, crc );
}



//////////////////////////////////////////////////////////////////////////
//
// Registration functions used at the beginning and end of each hkpDynamicsJob, and multi-threading registration functions.
//
//////////////////////////////////////////////////////////////////////////


void hkCheckDeterminismUtil::initThreadImpl()
{
	HK_THREAD_LOCAL_SET(m_inputStream, HK_NULL);
	HK_THREAD_LOCAL_SET(m_outputStream, HK_NULL);
	HK_THREAD_LOCAL_SET(m_jobInfoIdx, HK_NULL);
	HK_THREAD_LOCAL_SET(m_isPrimaryWorkerThread, HK_NULL);
	HK_THREAD_LOCAL_SET(m_currentJobFuid0, HK_NULL);
	HK_THREAD_LOCAL_SET(m_currentJobFuid1, HK_NULL);
	HK_THREAD_LOCAL_SET(m_currentJobFuid2, HK_NULL);
	HK_THREAD_LOCAL_SET(m_currentJobFuid3, HK_NULL);
}

void hkCheckDeterminismUtil::quitThreadImpl()
{
	HK_THREAD_LOCAL_SET(m_inputStream, HK_NULL);
	HK_THREAD_LOCAL_SET(m_outputStream, HK_NULL);
	HK_THREAD_LOCAL_SET(m_jobInfoIdx, HK_NULL);
	HK_THREAD_LOCAL_SET(m_isPrimaryWorkerThread, HK_NULL);
	HK_THREAD_LOCAL_SET(m_currentJobFuid0, HK_NULL);
	HK_THREAD_LOCAL_SET(m_currentJobFuid1, HK_NULL);
	HK_THREAD_LOCAL_SET(m_currentJobFuid2, HK_NULL);
	HK_THREAD_LOCAL_SET(m_currentJobFuid3, HK_NULL);
}

void hkCheckDeterminismUtil::workerThreadStartFrameImpl(hkBool isPrimaryWorkerThread)
{
	{
		hkUlong tmp = isPrimaryWorkerThread;
		HK_THREAD_LOCAL_SET(m_isPrimaryWorkerThread, reinterpret_cast<void*&>(tmp));
	}

	if (isPrimaryWorkerThread)
	{
		m_frame++;

		HK_ASSERT(0XAD876716, !m_writingStFromWorker);

		HK_ASSERT(0xad8766dd, m_primaryWorkerThreadInputStream == HK_NULL);
		HK_ASSERT(0xad8766dd, m_primaryWorkerThreadOutputStream == HK_NULL);

		if (getInstance().m_mode == hkCheckDeterminismUtil::MODE_COMPARE)
		{
			extractRegisteredJobsImpl();
		}

		// Create streams for single-threaded sections.
		registerAndStartJobImpl( Fuid::getZeroFuid() );

		void* tmp = HK_THREAD_LOCAL_GET(m_jobInfoIdx);
		m_primaryWorkerThreadJobInfoIdx = reinterpret_cast<int&>(tmp);
		m_primaryWorkerThreadInputStream  = HK_THREAD_LOCAL_GET(m_inputStream);
		m_primaryWorkerThreadOutputStream = HK_THREAD_LOCAL_GET(m_outputStream);

	}
	else
	{
		HK_THREAD_LOCAL_SET(m_inputStream, HK_NULL);
		HK_THREAD_LOCAL_SET(m_outputStream, HK_NULL);
	}
}

void hkCheckDeterminismUtil::workerThreadFinishFrameImpl()
{
	if (HK_THREAD_LOCAL_GET(m_isPrimaryWorkerThread))
	{
		HK_THREAD_LOCAL_SET(m_jobInfoIdx, reinterpret_cast<void*&>(m_primaryWorkerThreadJobInfoIdx));
		finishJob( Fuid::getZeroFuid(), false );

		m_primaryWorkerThreadInputStream = HK_NULL;
		m_primaryWorkerThreadOutputStream = HK_NULL;

		HK_ASSERT(0XAD876, !m_writingStFromWorker);

		if (hkCheckDeterminismUtil::getInstance().m_mode == hkCheckDeterminismUtil::MODE_WRITE)
		{
			hkCheckDeterminismUtil::getInstance().combineRegisteredJobs();
		}
		if (hkCheckDeterminismUtil::getInstance().m_mode == hkCheckDeterminismUtil::MODE_COMPARE)
		{
			hkCheckDeterminismUtil::getInstance().clearRegisteredJobs();
		}
	}
}


void hkCheckDeterminismUtil::delayJob(const Fuid& id, bool start) const
{
	if (m_delayJobs)
	{
		hkInt32 seed = hkInt32(m_frame) + hkInt32(start) + hkInt32(id.m_0) + hkInt32(id.getPackedJobId()) + hkInt32(id.m_2) + hkInt32(id.m_3) + hkInt32(id.m_4);
		unsigned int pixel = (seed*8) % (m_delayJobSeed.getSize()/4);
		const unsigned char* data = &m_delayJobSeed[pixel*4];
		hkUint32 cnt = data[0] ? m_delayCounter : 0;

		if (cnt)
		{
			//printf("delay fuid:%u, %u, %u, %u, %u \n", hkUint32(id.m_0), hkUint32(id.m_1), hkUint32(id.m_2), hkUint32(id.m_3), hkUint32(id.m_4));
			for (hkUint32 i=0; i<cnt; ++i)
			{
				setCurrentCheckIndex(i==0 ? 0 : -1);
			}
			setCurrentCheckIndex(0);
		}
		else
		{
			//printf("no delay fuid:%u, %u, %u, %u, %u \n", hkUint32(id.m_0), hkUint32(id.m_1), hkUint32(id.m_2), hkUint32(id.m_3), hkUint32(id.m_4));
		}
	}
}


void hkCheckDeterminismUtil::registerAndStartJobImpl(Fuid& jobFuid)
{
	m_shared->enter();

	if (HK_THREAD_LOCAL_GET(m_isPrimaryWorkerThread))
	{
		if (jobFuid == Fuid::getZeroFuid())
		{
			HK_ASSERT(0XAD98666D, HK_THREAD_LOCAL_GET(m_inputStream) == m_sharedInputStream);
			HK_ASSERT(0XAD98666D, HK_THREAD_LOCAL_GET(m_outputStream) == m_sharedOutputStream);

			HK_ASSERT(0XAD876, m_writingStFromWorker == false);
			m_writingStFromWorker = true;

		}
		else
		{
			HK_ASSERT(0XAD98666D, HK_THREAD_LOCAL_GET(m_inputStream) == m_primaryWorkerThreadInputStream);
			HK_ASSERT(0XAD98666D, HK_THREAD_LOCAL_GET(m_outputStream) == m_primaryWorkerThreadOutputStream);

			HK_ASSERT(0XAD876, m_writingStFromWorker );
			m_writingStFromWorker = false;
		}
	}
	else
	{
		HK_ASSERT(0XAD98666D, HK_THREAD_LOCAL_GET(m_inputStream) == HK_NULL);
		HK_ASSERT(0XAD98666D, HK_THREAD_LOCAL_GET(m_outputStream) == HK_NULL);
	}

	setCurrentJobFuid(jobFuid);
	setCurrentCheckIndex(0);

	// add/find entry to/in the shared list
	int i;
	for (i = 0; (i < m_registeredJobs.getSize()) && (m_registeredJobs[i].m_jobFuid != jobFuid); i++) {}

	// create local stream
	if (m_mode == MODE_WRITE)
	{
		// make sure it is unique
		if ( i != m_registeredJobs.getSize() )
		{
			HK_BREAKPOINT( 0xad9877da );		//"Internal error: Fuid is not frame-unique !!!"
		}
		int tmp = m_registeredJobs.getSize();
		HK_THREAD_LOCAL_SET(m_jobInfoIdx, reinterpret_cast<void*&>(tmp));
		JobInfo& info = m_registeredJobs.expandOne();
		info.m_jobFuid = jobFuid;
		info.m_isOpen = true;
		info.m_data = new hkMemoryTrack;
		HK_THREAD_LOCAL_SET(m_outputStream, new hkOstream(info.m_data));
	}
	else
	{
		HK_ASSERT2(0xad9877da, i < m_registeredJobs.getSize(), "Internal error: Fuid is not found !!!");
		HK_THREAD_LOCAL_SET(m_jobInfoIdx, reinterpret_cast<void*&>(i));
		// continue till the end of fthe m_registeredJobs, to ensure there is only one matching entry
		int j;
		for (j = i+1; (j < m_registeredJobs.getSize()) && (m_registeredJobs[j].m_jobFuid != jobFuid); j++) {}
		HK_ASSERT2(0xad9877da, j == m_registeredJobs.getSize(), "Internal error: Fuid is not frame-unique !!!");

		JobInfo& info = m_registeredJobs[i];
		HK_ASSERT2(0xad9877da, !info.m_isOpen, "Internal error: Fuid is not frame-unique !!!");
		info.m_isOpen = true;
		HK_THREAD_LOCAL_SET(m_inputStream, new hkIstream(info.m_data) );
	}

	hkCheckDeterminismUtil::checkMt(0xf0000000, 0xacacacacul);

	m_shared->leave();

	delayJob(jobFuid, true);
}

void hkCheckDeterminismUtil::finishJobImpl(Fuid& jobFuid, hkBool skipCheck)
{
	delayJob(jobFuid, false);

	m_shared->enter();
	// destroy local stream

	if( !skipCheck )
	{
		hkCheckDeterminismUtil::checkMt(0xf0000001, 0xbcbcbcbcul);
	}

	void* tmp = HK_THREAD_LOCAL_GET(m_jobInfoIdx);
	JobInfo& info = m_registeredJobs[reinterpret_cast<int&>(tmp)];
	HK_ASSERT2(0xad986dda, jobFuid == info.m_jobFuid, "Fuid inconsistency.");
	HK_ASSERT2(0xad9877db, info.m_isOpen, "Internal error: Fuid is not frame-unique !!!");
	info.m_isOpen = false;

	delete HK_THREAD_LOCAL_GET(m_inputStream);
	delete HK_THREAD_LOCAL_GET(m_outputStream);

	if (HK_THREAD_LOCAL_GET(m_isPrimaryWorkerThread))
	{
		hkIstream*  inputStream = HK_THREAD_LOCAL_GET( m_inputStream);
		hkOstream* outputStream = HK_THREAD_LOCAL_GET(m_outputStream);
		if ( (inputStream != m_primaryWorkerThreadInputStream) || (outputStream != m_primaryWorkerThreadOutputStream) )
		{
			HK_ASSERT2(0xad836433, getCurrentJobFuid() == jobFuid, "Job Fuid mismatch when finishing job.");
			HK_ASSERT2(0xad876333, (inputStream != m_sharedInputStream) || (outputStream != m_sharedOutputStream), "Trying to finish a job, but all the jobs are finished !!");

			HK_THREAD_LOCAL_SET(m_inputStream,  m_primaryWorkerThreadInputStream);
			HK_THREAD_LOCAL_SET(m_outputStream, m_primaryWorkerThreadOutputStream);

			HK_ASSERT(0XAD876, m_writingStFromWorker == false);
			m_writingStFromWorker = true;

			setCurrentJobFuid(Fuid::getZeroFuid());
		}
		else
		{
			HK_ASSERT2(0xad836433, jobFuid == Fuid::getZeroFuid(), "Special singleThreded job expected.");
			HK_ASSERT2(0xad836433, getCurrentJobFuid() == Fuid::getZeroFuid(), "Special singleThreded job expected.");
			HK_THREAD_LOCAL_SET(m_inputStream,  m_sharedInputStream);
			HK_THREAD_LOCAL_SET(m_outputStream, m_sharedOutputStream);

			HK_ASSERT(0XAD876, m_writingStFromWorker );
			m_writingStFromWorker = false;

			Fuid invalidFuid = Fuid::getZeroFuid();
			invalidFuid.m_0 = 0;
			setCurrentJobFuid(invalidFuid);
		}
	}
	else
	{
		HK_THREAD_LOCAL_SET(m_inputStream, HK_NULL);
		HK_THREAD_LOCAL_SET(m_outputStream, HK_NULL);

		Fuid invalidFuid = Fuid::getZeroFuid();
		invalidFuid.m_0 = 0;
		setCurrentJobFuid(invalidFuid);
	}

	m_shared->leave();
}

void hkCheckDeterminismUtil::cancelJobImpl(Fuid& jobFuid)
{
	m_shared->enter();

	int i;
	for (i = 0; (i < m_registeredJobs.getSize()) && (m_registeredJobs[i].m_jobFuid != jobFuid); i++) {}

	HK_ASSERT2( 0xe0173428, i < m_registeredJobs.getSize(), "Could not find job in registered job queue" );

	JobInfo& info = m_registeredJobs[i];

	if( m_mode == MODE_WRITE )
	{
		info.m_jobFuid = Fuid::getCanceledFuid();
	}

	m_shared->leave();
}

void hkCheckDeterminismUtil::combineRegisteredJobsImpl() // only used on write
{
	HK_TIMER_BEGIN("hkCheckDeterminismUtil::combineRegisteredJobsImpl", HK_NULL);

	HK_ASSERT(0xad8666dd, m_registeredJobs[0].m_jobFuid == Fuid::getZeroFuid());

	m_shared->enter();

	HK_ASSERT(0XAD876, !m_writingStFromWorker);
	HK_ASSERT(0XAD876655, m_primaryWorkerThreadInputStream == HK_NULL);
	HK_ASSERT(0XAD876655, m_primaryWorkerThreadOutputStream == HK_NULL);

	hkUint32 check = hkUint32(0xadadadad);
	m_sharedOutputStream->write((char*)&check, sizeof(check));

	//First, remove any jobs which were cancelled
	for( int i = 0; i < m_registeredJobs.getSize(); i++ )
	{
		if( m_registeredJobs[i].m_jobFuid == Fuid::getCanceledFuid() )
		{
			delete m_registeredJobs[i].m_data;
			m_registeredJobs.removeAt( i );
		}
	}

	// header; jobs count
	int numRegisteredJobs = m_registeredJobs.getSize();
	m_sharedOutputStream->write((char*)&numRegisteredJobs, sizeof(numRegisteredJobs));

	// combine streams from registered jobs
	{
		for (int i = 0; i < m_registeredJobs.getSize(); i++)
		{
			JobInfo& info = m_registeredJobs[i];
			Fuid fuid = info.m_jobFuid;
			m_sharedOutputStream->write((char*)&fuid, sizeof(fuid));
			int dataSize = info.m_data->getSize();
			m_sharedOutputStream->write((char*)&dataSize, sizeof(dataSize));

			if ( m_memoryTrack )
			{
				m_memoryTrack->appendByMove( info.m_data );
			}
			else
			{
				char buffer[2048];				
				while ( dataSize > 0)
				{
					int numBytes = hkMath::min2( dataSize, 2048 );
					info.m_data->read( buffer, numBytes );
					m_sharedOutputStream->write(buffer, numBytes);
					dataSize -= numBytes;				
				}
			}
			HK_ASSERT2(0XAD8765dd, !info.m_isOpen, "Job not finished.");
			delete info.m_data;
		}
		m_registeredJobs.clear();
		m_registeredJobs.reserveExactly(0);
	}

	{
		check = hkUint32(0xbdbdbdbd);
		m_sharedOutputStream->write((char*)&check, sizeof(check));
	}

	m_shared->leave();
	HK_TIMER_END();

}

void hkCheckDeterminismUtil::extractRegisteredJobsImpl() // only used for read
{
	HK_TIMER_BEGIN("hkCheckDeterminismUtil::extractRegisteredJobsImpl", HK_NULL);
	m_shared->enter();

	HK_ASSERT(0XAD876, !m_writingStFromWorker);

	HK_ASSERT(0XAD876655, m_primaryWorkerThreadInputStream == HK_NULL);
	HK_ASSERT(0XAD876655, m_primaryWorkerThreadOutputStream == HK_NULL);

	hkUint32 check;
	m_sharedInputStream->read((char*)&check, sizeof(check));
	HK_ASSERT2(0xad8655dd, check == hkUint32(0xadadadad), "Stream inconsistent.");

	HK_ASSERT2(0xad87656d, m_registeredJobs.getSize() == 0, "Internal inconsistency.");

	int numRegisteredJobs;
	m_sharedInputStream->read((char*)&numRegisteredJobs, sizeof(numRegisteredJobs));

	for (int i = 0; i < numRegisteredJobs; i++)
	{
		JobInfo& info = m_registeredJobs.expandOne();
		info.m_isOpen = false;
		info.m_data = new hkMemoryTrack;

		m_sharedInputStream->read((char*)&info.m_jobFuid, sizeof(Fuid));

		int dataSize;
		m_sharedInputStream->read((char*)&dataSize, sizeof(dataSize));

		// read the data
		while ( dataSize > 0 )
		{
			int numBytes = hkMath::min2( dataSize, 2048 );
			char buffer[2048];
			m_sharedInputStream->read(buffer, numBytes);
			info.m_data->write( buffer, numBytes );
			dataSize -= numBytes;
		}
	}

	m_sharedInputStream->read((char*)&check, sizeof(check));
	HK_ASSERT2(0xad8655d1, check == hkUint32(0xbdbdbdbd), "Stream inconsistent.");

	HK_ASSERT(0XAD876, !m_writingStFromWorker);

	m_shared->leave();
	HK_TIMER_END();
}

void hkCheckDeterminismUtil::clearRegisteredJobsImpl()
{
	m_shared->enter();

	HK_ASSERT(0XAD876, !m_writingStFromWorker);
	HK_ASSERT(0XAD876655, m_primaryWorkerThreadInputStream == HK_NULL);
	HK_ASSERT(0XAD876655, m_primaryWorkerThreadOutputStream == HK_NULL);

	// destroy streams from individual jobs
	for (int i = 0; i < m_registeredJobs.getSize(); i++)
	{
		JobInfo& info = m_registeredJobs[i];
		HK_ASSERT2(0XAD8765dd, !info.m_isOpen, "Job not finished.");
		delete info.m_data;
	}
	m_registeredJobs.clear();
	m_registeredJobs.reserveExactly(0);

	m_shared->leave();

}

void hkCheckDeterminismUtil::setCurrentJobFuid(hkCheckDeterminismUtil::Fuid jobFuid)
{
	HK_ASSERT2(0xad342843, sizeof(jobFuid) == 16, "Job Fuid is greater than 3 bytes. Adjust the code.");
	hkUint32* fuidPtr = (hkUint32*)&jobFuid;
	HK_THREAD_LOCAL_SET(m_currentJobFuid0, fuidPtr[0]);
	HK_THREAD_LOCAL_SET(m_currentJobFuid1, fuidPtr[1]);
	HK_THREAD_LOCAL_SET(m_currentJobFuid2, fuidPtr[2]);
	HK_THREAD_LOCAL_SET(m_currentJobFuid3, fuidPtr[3]);
	setCurrentCheckIndex(hkUint32(-1));

	if (m_enableThreadTracker)
	{
		int ti = HK_THREAD_LOCAL_GET( hkThreadNumber );
		if (ti >= m_threadTracker.getSize())
			m_threadTracker.setSize(ti+1);
		m_threadTracker[ti] = jobFuid;
		m_maxTheadId = ti > m_maxTheadId ? ti : m_maxTheadId;
	}
}

hkUint32 hkCheckDeterminismUtil::getCurrentCheckIndex()
{
	hkUint32 ci = HK_THREAD_LOCAL_GET(m_currentCheckIndex);
	return ci;
}


void hkCheckDeterminismUtil::setCurrentCheckIndex(hkUint32 checkIndex)
{
	HK_THREAD_LOCAL_SET(m_currentCheckIndex, checkIndex);
}

void hkCheckDeterminismUtil::bumpCurrentCheckIndex()
{
	hkUint32 checkIndex = HK_THREAD_LOCAL_GET(m_currentCheckIndex);
	HK_THREAD_LOCAL_SET(m_currentCheckIndex, checkIndex+1);
}

hkCheckDeterminismUtil::Fuid hkCheckDeterminismUtil::getCurrentJobFuid()
{
	Fuid result;
	hkUint32* fuidPtr = (hkUint32*)&result;
	fuidPtr[0] = HK_THREAD_LOCAL_GET(m_currentJobFuid0);
	fuidPtr[1] = HK_THREAD_LOCAL_GET(m_currentJobFuid1);
	fuidPtr[2] = HK_THREAD_LOCAL_GET(m_currentJobFuid2);
	fuidPtr[3] = HK_THREAD_LOCAL_GET(m_currentJobFuid3);
	return result;
}

void HK_CALL hkCheckDeterminismUtil::wipeStackImpl( hkUint32 value )
{
	// 512 bytes are wiped
	const int wipeSize = 128;
	hkUint32 array[ wipeSize ];
	hkString::memSet4( array, value, wipeSize );
}

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
