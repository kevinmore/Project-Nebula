/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#ifndef HK_BASE_WIISTREAMWRITER_H
#define HK_BASE_WIISTREAMWRITER_H

#include <Common/Base/System/Io/Writer/hkStreamWriter.h>

class hkSocket;

class hkWiiStreamWriter : public hkStreamWriter
{
public:

	hkWiiStreamWriter (const char* fname);

	void close();
	virtual ~hkWiiStreamWriter ();
	virtual int write( const void* buf, int nbytes);
	virtual hkBool isOk() const;
	virtual hkBool seekTellSupported() const;
	virtual hkResult seek( int offset, SeekWhence whence);
	virtual int tell() const;

	// hkWiiStreamWriter only becomes active if enabled first via the static member function 
	// hkWiiStreamWriter::enable() (in the demo framework, this is done in hkFrameworkMain).
	// This is because the visual debugger and the file writer have to use the same HIO2 buffers, 
	// so one should be used only if the other is not.
	static void enable();

protected:

	void open( const char* nameIn );
	void _readCompletionPacket();
	
	hkSocket* m_socket;
	static hkSocket* m_server;

	const int m_maxDataPerPacket;	
	bool m_isOpen;
	
	static bool isEnabled;
	
};

#endif //HK_BASE_WIISTREAMWRITER_H

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
