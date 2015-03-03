/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/System/Io/Writer/Buffered/hkBufferedStreamWriter.h>

HK_SINGLETON_MANUAL_IMPLEMENTATION(hkError);

hkErrStream::hkErrStream( void* buf, int bufSize )
: hkOstream( (hkStreamWriter*)HK_NULL)
{
	int sizeOfWriter = HK_NEXT_MULTIPLE_OF(16,sizeof(hkBufferedStreamWriter));
	void* p = ((char*)buf) + bufSize - sizeOfWriter;
	m_writer = new (p) hkBufferedStreamWriter(buf, bufSize - sizeOfWriter, true);
	m_writer->addReferenceLockUnchecked();
}

extern void HK_CALL hkReferenceCountError(const hkReferencedObject*, const char* why);

void HK_CALL hkReferenceCountError(const hkReferencedObject* o, const char* why)
{
	HK_ERROR(0x2c66f2d8, "Reference count error on object " << (const void*)(o)
			<< " with ref count of " << o->getReferenceCount()
			<< " in " << why << ".\n"
			<< " * Are you calling delete instead of removeReference?\n"
			<< " * Have you called removeReference too many times?\n"
			<< " * In a multithreaded environment, what is the hkReferencedObject lock mode you use (see setLockMode())?\n"
			<< " * Is this a valid object?\n"
			<< " * Do you have more than 32768 references? (unlikely)\n");
}

extern "C" void HK_CALL hkErrorMessage( const char* c );

void HK_CALL hkErrorMessage( const char* c )
{
	HK_ERROR(0x2636fe25, c);
}

int HK_CALL hkError::messageReport(int id, const char* description, const char* file, int line)
{
	return hkError::getInstance().message(MESSAGE_REPORT, id, description, file, line);
}

int HK_CALL hkError::messageWarning(int id, const char* description, const char* file, int line)
{
	return hkError::getInstance().message(MESSAGE_WARNING, id, description, file, line);
}

int HK_CALL hkError::messageAssert(int id, const char* description, const char* file, int line)
{
	return hkError::getInstance().message(MESSAGE_ASSERT, id, description, file, line);
}

int HK_CALL hkError::messageError(int id, const char* description, const char* file, int line)
{
	return hkError::getInstance().message(MESSAGE_ERROR, id, description, file, line);
}


/* Asserts Id's for use with HK_ASSERT2
 * Pick an ID, use it and delete it from the list below

0x6569eafd
0x7e9ce7c5
0x21be44cd
0x2f718942
0x63482c0a
0x1b87e64c
0x7f47a720
0x58f928ef
0x4fda0b4a
0x5b6c0e14
*/

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
