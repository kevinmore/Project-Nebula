/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#if ! defined (HK_PLATFORM_SPU)
inline void hkSolverExportWriteRhs(hkSimdRealParameter rhs, hkpSolverResults* solverResultInMainMemory)
{
	rhs.store<1>(&solverResultInMainMemory[0].m_internalSolverData);
}
inline void hkSolverExportWriteOneResult(hkVector4Parameter imp_rhs, hkpSolverResults* solverResultInMainMemory, class hkSpuDmaSparseWriter* writer)
{
	// expect vector to contain impulseApplied, rhs
	imp_rhs.store<2,HK_IO_NATIVE_ALIGNED>(&solverResultInMainMemory[0].m_impulseApplied);
}
inline void hkSolverExportWriteOneResult(hkSimdRealParameter impulseApplied, hkSimdRealParameter rhs, hkpSolverResults* solverResultInMainMemory, class hkSpuDmaSparseWriter* writer)
{
	impulseApplied.store<1,HK_IO_NATIVE_ALIGNED>(&solverResultInMainMemory[0].m_impulseApplied);
	rhs.store<1,HK_IO_NATIVE_ALIGNED>(&solverResultInMainMemory[0].m_internalSolverData);
}
inline void hkSolverExportWriteTwoResults(hkVector4Parameter imp_rhs, hkpSolverResults* solverResultInMainMemory, class hkSpuDmaSparseWriter* writer)
{
	// expect vector to contain impulseApplied0, rhs0, impulseApplied1, rhs1
	// therefore check that the two solverResultInMainMemory are contiguous
	HK_ASSERT2(0x384ff2bb, ((hkUlong)&solverResultInMainMemory[1].m_impulseApplied) - ((hkUlong)&solverResultInMainMemory[0].m_impulseApplied) == (hkUlong)(sizeof(hkReal)*2), "solver results memory not contiguous");
	imp_rhs.store<4,HK_IO_NATIVE_ALIGNED>(&solverResultInMainMemory[0].m_impulseApplied);
}
#else
#include <Common/Base/Spu/Dma/SparseWriter/hkSpuDmaSparseWriter.h>

inline void hkSolverExportWriteOneResult(hkReal impulseApplied, hkReal rhs, hkpSolverResults* solverResultInMainMemory, hkSpuDmaSparseWriter* writer)
{
	writer->putToMainMemorySmall64(impulseApplied, rhs, solverResultInMainMemory);
}
inline void hkSolverExportWriteOneResult(hkVector4Parameter imp_rhs, hkpSolverResults* solverResultInMainMemory, hkSpuDmaSparseWriter* writer)
{
	writer->putToMainMemorySmall64(imp_rhs(0), imp_rhs(1), solverResultInMainMemory);
}
inline void hkSolverExportWriteOneResult(hkSimdRealParameter impulseApplied, hkSimdRealParameter rhs, hkpSolverResults* solverResultInMainMemory, class hkSpuDmaSparseWriter* writer)
{
	writer->putToMainMemorySmall64(impulseApplied.getReal(), rhs.getReal(), solverResultInMainMemory);
}
inline void hkSolverExportWriteTwoResults(hkVector4Parameter imp_rhs, hkpSolverResults* solverResultInMainMemory, hkSpuDmaSparseWriter* writer)
{
	writer->putToMainMemorySmall128(imp_rhs(0), imp_rhs(1), imp_rhs(2), imp_rhs(3), solverResultInMainMemory);
}
#endif

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
