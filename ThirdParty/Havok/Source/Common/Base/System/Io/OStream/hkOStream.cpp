/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/System/Io/OStream/hkOStream.h>
#include <Common/Base/Container/String/hkStringBuf.h>

#include <Common/Base/Fwd/hkcstdarg.h>
#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/Reflection/hkTypeInfo.h>
#include <Common/Base/System/Io/FileSystem/hkFileSystem.h>
#include <Common/Base/System/Io/Writer/Array/hkArrayStreamWriter.h>
#include <Common/Base/System/Io/Writer/Buffered/hkBufferedStreamWriter.h>

using namespace std;

static const int HK_BUFSIZ = 10024;

#ifndef HK_PLATFORM_SPU
HK_REFLECTION_DEFINE_STUB_VIRTUAL_BASE(hkOstream);
#endif

static void HK_CALL writeString(hkStreamWriter* sb, const char* s)
{
	if(s)
	{
		sb->write( s, hkString::strLen(s) );
	}
	else
	{
		sb->write( "(null)", 6);
	}
}


hkOstream::hkOstream(hkStreamWriter* sw)
	:	m_writer(sw)
{
}

hkOstream::hkOstream(const char* filename)
{
	m_writer = hkFileSystem::getInstance().openWriter(filename);
}

hkOstream::hkOstream(void* mem, int memSize, hkBool isString)
{
	m_writer.setAndDontIncrementRefCount( new hkBufferedStreamWriter(mem, memSize, isString) );
}

hkOstream::hkOstream( hkArray<char, hkContainerHeapAllocator>& buf )
{
	m_writer.setAndDontIncrementRefCount( new hkArrayStreamWriter(&buf, hkArrayStreamWriter::ARRAY_BORROW) );
}

hkOstream::hkOstream( hkMemoryTrack* buf )
{
	m_writer.setAndDontIncrementRefCount( new hkMemoryTrackStreamWriter( buf, hkMemoryTrackStreamWriter::TRACK_BORROW ) );
}


hkOstream::~hkOstream()
{
}

hkBool hkOstream::isOk() const
{
	return m_writer && m_writer->isOk();
}

hkOstream& hkOstream::operator<< (const void* p)
{
	char buf[HK_BUFSIZ];
	hkString::snprintf(buf, HK_BUFSIZ, "%p", p);
	writeString(m_writer, buf);
	return *this;
}

hkOstream& hkOstream::operator<< (hkBool b)
{
	writeString(m_writer, b ? "true" : "false" );
	return *this;
}

hkOstream& hkOstream::operator<< (char c)
{
	m_writer->write(&c, 1);
	return *this;
}

hkOstream& hkOstream::operator<< (const char* s)
{
	writeString(m_writer, s);
	return *this;
}

hkOstream& hkOstream::operator<< (short s)
{
	char buf[HK_BUFSIZ];
	hkString::snprintf(buf, HK_BUFSIZ, "%i", s);
	writeString(m_writer, buf);
	return *this;
}

hkOstream& hkOstream::operator<< (unsigned short s)
{
	char buf[HK_BUFSIZ];
	hkString::snprintf(buf, HK_BUFSIZ, "%u", s);
	writeString(m_writer, buf);
	return *this;
}

hkOstream& hkOstream::operator<< (int i)
{
	char buf[HK_BUFSIZ];
	hkString::snprintf(buf, HK_BUFSIZ, "%i", i);
	writeString(m_writer, buf);
	return *this;
}

hkOstream& hkOstream::operator<< (unsigned int u)
{
	char buf[HK_BUFSIZ];
	hkString::snprintf(buf, HK_BUFSIZ, "%u", u);
	writeString(m_writer, buf);
	return *this;
}

hkOstream& hkOstream::operator<< (const hkSimdFloat32& f)
{
	char buf[HK_BUFSIZ];
	hkString::snprintf(buf, HK_BUFSIZ, "%f", f.getReal());
	writeString(m_writer, buf);
	return *this;
}

hkOstream& hkOstream::operator<< (const hkSimdDouble64& f)
{
	char buf[HK_BUFSIZ];
	hkString::snprintf(buf, HK_BUFSIZ, "%f", f.getReal());
	writeString(m_writer, buf);
	return *this;
}


hkOstream& hkOstream::operator<< (float f)
{
	char buf[HK_BUFSIZ];
	hkString::snprintf(buf, HK_BUFSIZ, "%f", f);
	writeString(m_writer, buf);
	return *this;
}

hkOstream& hkOstream::operator<< (hkInt64 i)
{
	char buf[HK_BUFSIZ];
	hkString::snprintf(buf, HK_BUFSIZ, HK_PRINTF_FORMAT_INT64, i);
	writeString(m_writer, buf);
	return *this;
}
hkOstream& hkOstream::operator<< (hkUint64 i)
{
	char buf[HK_BUFSIZ];
	hkString::snprintf(buf, HK_BUFSIZ, HK_PRINTF_FORMAT_UINT64, i);
	writeString(m_writer, buf);
	return *this;
}

hkOstream&  hkOstream::operator<< (const hkVector4f& v)
{
	this->printf("[%g,%g,%g,%g]", v(0), v(1), v(2), v(3));
	return *this;
}

hkOstream&  hkOstream::operator<< (const hkVector4d& v)
{
	this->printf("[%g,%g,%g,%g]", v(0), v(1), v(2), v(3));
	return *this;
}



hkOstream&  hkOstream::operator<< (const hkQuaternionf& q)
{
	this->printf("[%f,%f,%f,(%f)]", q.m_vec(0), q.m_vec(1), q.m_vec(2), q.m_vec(3));
	return *this;
}

hkOstream&  hkOstream::operator<< (const hkQuaterniond& q)
{
	this->printf("[%f,%f,%f,(%f)]", q.m_vec(0), q.m_vec(1), q.m_vec(2), q.m_vec(3));
	return *this;
}

hkOstream&  hkOstream::operator<< (const hkMatrix3f& m)
{
	for( int i=0; i<3; ++i)
	{
		this->printf("|%f,%f,%f|\n", m(i,0), m(i,1), m(i,2) );
	}
	return *this;
}

hkOstream&  hkOstream::operator<< (const hkMatrix3d& m)
{
	for( int i=0; i<3; ++i)
	{
		this->printf("|%f,%f,%f|\n", m(i,0), m(i,1), m(i,2) );
	}
	return *this;
}

hkOstream&  hkOstream::operator<< (const hkTransformf& t)
{
	return (*this) << t.getRotation() << t.getTranslation();
}

hkOstream&  hkOstream::operator<< (const hkTransformd& t)
{
	return (*this) << t.getRotation() << t.getTranslation();
}



void hkOstream::printf(const char *fmt, ...)
{
	char buf[HK_BUFSIZ];
	va_list args; 
	va_start(args, fmt);
	hkString::vsnprintf(buf, HK_BUFSIZ, fmt, args);
	va_end(args);
	writeString(m_writer, buf);
}

hkOstream& hkOstream::operator<< (const hkStringPtr& str)
{
	if (str.cString())
	{
		m_writer->write( str.cString(), str.getLength() ); 
	}
	else
	{
		m_writer->write( "(null)", 6 ); 
	}
	return *this;
}

hkOstream& hkOstream::operator<< (const hkStringBuf& str)
{
	if (str.cString())
	{
		m_writer->write( str.cString(), str.getLength() ); 
	}
	else
	{
		m_writer->write( "(null)", 6 ); 
	}
	return *this;
}

void hkOstream::flush()
{
	m_writer->flush();
}

// Output raw data.
int hkOstream::write( const char* buf, int nbytes)
{
	return m_writer->write(buf, nbytes);
}

void hkOstream::setStreamWriter(hkStreamWriter* newWriter)
{
	m_writer = newWriter;
}

#ifndef HK_PLATFORM_SPU
const hkClass* hkOstream::getClassType() const
{ 
	return &hkOstreamClass; 
}
#endif

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
