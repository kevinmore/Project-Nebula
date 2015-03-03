/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKGP_CONVEX_DECOMPOSITION_H
#define HKGP_CONVEX_DECOMPOSITION_H

#include <Common/Base/hkBase.h> // Xcode thinks thus is the auto pch as it is the project name 
#include <Common/Base/Types/Geometry/Aabb/hkAabb.h>

#include <Common/Internal/GeometryProcessing/ConvexHull/hkgpConvexHull.h>
struct	hkGeometry;

///
/// Convex decomposition
///
struct hkgpConvexDecomposition
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE,hkgpConvexDecomposition);
	///
	/// Progress interface
	///
	struct IProgress
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE,hkgpConvexDecomposition::IProgress);
		virtual			~IProgress() {}
		
		/// Start a decomposition job, return false to stop processing.
		virtual bool	startJob(int index,int totalJobs) const=0;
		
		/// Step 1, decompose planes into convex sets, return false to stop processing.
		virtual bool	onDecomposePlanes(int currentPlane,int totalPlanes) const=0;

		/// Step 2, reduce convex hulls, return false to stop processing.
		virtual bool	onReduce(int currentHullCount,int totalHulls) const=0;

		/// Step 3, decimate convex hulls, return false to stop processing.
		virtual bool	onDecimate(int currentHullCount,int objectiveHullsCount) const=0;

		/// End of a decomposition job.
		virtual void	endJob() const=0;
	};
	
	///
	/// Data attached to convex hulls
	///
	struct AttachedData : public hkgpConvexHull::IUserObject
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY,AttachedData);
											AttachedData() : m_materialID(-1) {}
		inline hkgpConvexHull::IUserObject*	clone() const										{ return new AttachedData(*this); }
		inline void							append(const hkgpConvexHull::IUserObject* iOther)	{ HK_ERROR(0x74B5683C,"Not allowed to compound"); }
		
		int					m_materialID;					///< Material ID from the original geometry.
	};
	
	///
	/// Guard generator config
	///
	struct GuardGenConfig
	{
		/// Generation method
		enum eMethod
		{
			NONE,			///< Do not generate guards.
			SPHERE,			///< Generate sphere to prevent merging.
			EDGE,			///< Generate edge to prevent merging.
		};
		GuardGenConfig();

		eMethod					m_method;						///< Generation method.
		hkReal					m_offset;						///< Offset to the surface.
		hkReal					m_edgeSamplingSpacing;			///< Spacing between edge samples (for SPHERE method only).
		hkBool32				m_useVertexWeight;				///< Use offset * vertex.W as actual offset.
		
		hkReal					m_globalGuardsScale;			///< Scale applied to global (Octree) guards.
		int						m_maxGlobalGuardsOctreeDepth;	///< Maximum depth allowed for global guards generation.
	};

	///
	/// Mesh pre-processor config
	///
	struct MeshPreProcessorConfig
	{
		MeshPreProcessorConfig();

		hkBool				m_optimize;						///< Apply Collision Geometry Optimize (CGO) to the mesh.
		hkBool				m_simplifyPlanar;				///< Simplify planar faces.
		hkBool				m_repairTJunctions;				///< Attempt to automatically repair T-junctions.
		hkBool				m_closeHoles;					///< Attempt to automatically close geometry holes.

		hkReal				m_planeCosAngle;				///< Maximum cosine of the angle between triangles used to classify planes.
		hkReal				m_maximumTJunctionsDistance;	///< Distance below which T-junctions gets repaired.
		hkReal				m_maximumOptimizationError;		///< Maximum allowed error during optimization (unit: distance).
	};

	///
	/// Configuration
	///
	struct Config
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE,hkgpConvexDecomposition::Config);
		/// Primary decomposition method.
		enum ePrimaryMethod
		{
			SURFACE,					///< Surface (default).
			SOLID,						///< Solid.
			WRAP,						///< Sequential hull wrapping.
			DELAUNAY,					///< Delaunay [experimental].
			NUM_PRIMARY_METHODS
		};
		
		/// Reduce method.
		enum eReduceMethod
		{
			REDUCE_DISABLE,				///< Reduce step disable.
			REDUCE_LOCAL,				///< Reduce step restricted to local merges.
			REDUCE_GLOBAL,				///< Reduce step allowed for global merges.
			NUM_REDUCE_METHODS
		};

		/// Hollows parts handling.
		enum eHollowParts
		{
			HOLLOW_KEEP,				///< Process hollows as disconnected parts.
			HOLLOW_MERGE,				///< Merge hollows with their outer shell.
			HOLLOW_DISCARD,				///< Disregard hollows.
			NUM_HOLLOW_METHODS
		};
		
		Config();

		/* Basic settings					*/ 
		hkEnum<ePrimaryMethod,hkInt32>	m_primaryMethod;				///< Primary decomposition method.
		hkEnum<eReduceMethod,hkInt32>	m_reduceMethod;					///< Reduce method.
		hkEnum<eHollowParts,hkInt32>	m_hollowMethod;					///< Hollow handling method.

		/* Basic accuracy settings			*/ 
		hkReal							m_accuracy;						///< Fraction of total number of convex hulls to retain after 'exact' decomposition.
		hkInt32							m_maxPieces;					///< Maximum pieces to generate.

		/* Wrap settings					*/ 
		hkInt32							m_maxWrapIterations;			///< Maximum iterations allowed for WRAP primary method.
		hkReal							m_maxWrapConcacity;				///< Iterate until concavity is lower than this value (or maxWrapIterations is reached).

		/* Basic guards related settings	*/ 
		GuardGenConfig					m_guardsGenerator;				///< Guards generator config.
		hkArray<hkVector4>				m_sphereGuards;					///< Sphere guards list as [position(xyz),radius(w)]*.
		hkArray<hkVector4>				m_edgeGuards;					///< Edge guards list as [start,end(xyz),radius(w)]*.
		
		/* Overlaps solver settings			*/ 
		hkInt32							m_reduceOverlapsPasses;			///< Try to reduce again if overlaps are found during the reduction step.
		hkInt32							m_decimateOverlapsPasses;		///< Try to decimate again if overlaps are found during the decimation step.
		hkInt32							m_finalOverlapsPasses;			///< Number of final overlaps resolution passes.
		hkReal							m_finalOVerlapsMinVolume;		///< Only overlapping volume greater than this number will be resolved.

		/* Output control					*/ 
		hkReal							m_volumeSimplification;			///< Maximum volume change for final convex hull simplification (disabled if use existing is true).
		hkInt32							m_maxVertices;					///< Maximum number of vertices per convex hull.
		
		/* Advanced settings				*/ 
		hkReal							m_maxCosAngle;					///< Cosine of the angle above which two triangles are classified as coplanar.
		hkReal							m_maxDistance;					///< Maximum distance to plane below which two triangles are classified as coplanar.
		hkReal							m_maxPrjPlaneCosAngle;			///< Cosine of the angle above which two convex hull projection planes are classified as coplanar.
		hkReal							m_maxPenetration;				///< Penetration depth allowed for exact decomposition.
		hkReal							m_minThickness;					///< Minimum thickness allowed for surface elements (0 for 2D/planar).
		hkReal							m_minHullWidth;					///< Minimum allowed convex hulls width.
		hkReal							m_minOverlapVolume;				///< Minimum operand volume required to process overlaps.
		hkReal							m_maxDelaunayMergeError;		///< Maximum error allowed for merge when using ePrimaryMethod::DELAULAY method.
		hkBool							m_mergeAllParts;				///< Merge all parts into one.
		hkBool							m_useMaterialBoundaries;		///< Do not merge convex pieces with different materials.
		hkInt32							m_numThreads;					///< Number of threads to use.
		hkInt32							m_maxCellsPerDomain;			///< Maximum cells per domain, used for solid decomposition.
		hkInt32							m_verbosity;					///< Reports verbosity.
		MeshPreProcessorConfig			m_meshPreProcessor;				///< Mesh pre-processor config.
		const IProgress*				m_iprogress;					///< Progress interface pointer.
		const void*						m_internal;						///< Internal pointer [INTERNAL].
	};
	
	//
	// Methods
	//

	/// Decompose to a convex hull set or decimate an existing set if hullsInOut is not empty.
	static void	HK_CALL		decompose(const Config& config, const hkGeometry& geometry, hkArray<hkgpConvexHull*>& hullsInOut);

	/// Decompose to a convex hull set or decimate an existing set if hullsInOut is not empty.
	static void	HK_CALL		decompose(const Config& config, const hkGeometry& geometry, hkArray<hkArray<hkVector4> >& hullsInOut);
	
	/// Generate sphere guards.
	static void HK_CALL		generateSphereGuards(const hkGeometry& geometry, hkReal offset, hkReal samplingRate, hkArray<hkVector4>& sphereGuardsOut);

	/// Generate edge guards.
	static void HK_CALL		generateEdgeGuards(const hkGeometry& geometry, hkReal offset, hkBool32 useVertexWeights, hkArray<hkVector4>& edgeGuardsOut, int numThreads=1);
};

#endif // HKGP_CONVEX_DECOMPOSITION_H

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
