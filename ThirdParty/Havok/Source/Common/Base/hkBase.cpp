/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>

// The following whitespace is important. DON'T DELETE
// This is to offset the lines of the compile-time asserts
// so that they don't clash with other asserts in other files when we do
// the uber build
//
//
//
//
//
//
//
//
// End of Whitespace



HK_COMPILE_TIME_ASSERT( sizeof(hkInt8) == 1 );
HK_COMPILE_TIME_ASSERT( sizeof(hkUint8) == 1 );
HK_COMPILE_TIME_ASSERT( sizeof(hkInt16) == 2 );
HK_COMPILE_TIME_ASSERT( sizeof(hkUint16) == 2 );  
HK_COMPILE_TIME_ASSERT( sizeof(hkInt32) == 4 );
HK_COMPILE_TIME_ASSERT( sizeof(hkUint32) == 4 );
HK_COMPILE_TIME_ASSERT( sizeof(hkInt64) == 8 );
HK_COMPILE_TIME_ASSERT( sizeof(hkUint64) == 8 );
HK_COMPILE_TIME_ASSERT( sizeof(hkUlong) == sizeof(void*) );

/*
#if defined(HK_MATH_VECTOR4_UTIL_H)
	#error hkBase.h should not include hkVector4Util.h please fix
#endif

#if defined(HK_MATH_INTVECTOR_H)
	#error	hkBase.h should not include hkIntVector.h please fix
#endif
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
