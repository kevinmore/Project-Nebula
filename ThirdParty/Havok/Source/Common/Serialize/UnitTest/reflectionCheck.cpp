/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>

#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/System/Io/Writer/hkStreamWriter.h>
#include <Common/Serialize/Util/hkBuiltinTypeRegistry.h>
#include <Common/Serialize/Util/hkSerializationCheckingUtils.h>

namespace 
{
	struct NullStream : public hkStreamWriter
	{
		int write(const void* buf, int nbytes)	{ return nbytes; }
		hkBool isOk() const { return true; }
	};
}

static int reflectionCheck()
{
#ifdef HK_REAL_IS_DOUBLE
	HK_REPORT("This unit test is disabled in DP builds, as reflection system cannot distinguish 32 and 64 Bit floating point numbers.");
#else
	HK_TEST( 1+1 == 2 );
	hkSerializationCheckingUtils::DeferredErrorStream deferred;
	hkOstream output(&deferred);

	hkResult res;
	hkError::getInstance().setEnabled(0x786cb087, false);
	const hkClassNameRegistry* classReg = hkBuiltinTypeRegistry::getInstance().getClassNameRegistry();
	HK_ASSERT(0x3bcd08e0, classReg);
	const char* memoryManagedPrefixes[] = { "hcl" };
	const int numPrefixes = (int) HK_COUNT_OF(memoryManagedPrefixes);
	bool reportNonMemoryManaged = true;
	res = hkSerializationCheckingUtils::verifyReflection(*classReg, output, memoryManagedPrefixes, numPrefixes, reportNonMemoryManaged );

	if( res != HK_SUCCESS )
	{
		deferred.dump();
	}
	HK_TEST( res == HK_SUCCESS );

	hkError::getInstance().setEnabled(0x786cb087, true);
#endif
	return 0;
}


#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(reflectionCheck, "Fast", "Common/Test/UnitTest/Serialize/", __FILE__     );

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
