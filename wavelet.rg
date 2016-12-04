import "regent"

-- FIXME: Remove token dependencies
-- FIXME: Check if order actually matters in the outer while loop?
-- FIXME: Can x and y things be done completely independently? I don't think
-- so - but check it by modifiying c++ code.
--
-- FIXME: Because we have two partitions, should we give each num_parallelism number
-- of partitions, or just divide those in half?

-- FIXME: spmd style programming.
-- Differences:
--    He runs on all three channels (RGB) separately.
--

-- Helper modules to handle PNG files and command line arguments
local png        = require("png_util")
local WaveletConfig = require("wavelet_config")

-- Some C APIs
local c     = regentlib.c
local sqrt  = regentlib.sqrt(double)
local cmath = terralib.includec("math.h")
local PI = cmath.M_PI
local unistd = terralib.includec("unistd.h")

terra wait_for(x : int) return 1 end

-- Field space for pixels
-- Should be the only required field because we make changes in place.
fspace Pixel
{
  value      : uint8;    -- value pixel in 8-bit gray scale
}

-- FIXME: Which parameters do we need to provide in comparison to gfxc?
--struct Parameters
--{
  --step : int32;
  --size : int2d;
--}

task initialize(r_image : region(ispace(int2d), Pixel),
                filename : rawstring)
where
  reads writes(r_image)
do
  png.read_png_file(filename,
                    __physical(r_image.value),
                    __fields(r_image.value),
                    r_image.bounds)
end

task liftX(r_image    : region(ispace(int2d), Pixel), 
          step : int32)
where
  reads writes(r_image.value)
do 
  var ts_start = c.legion_get_current_time_in_micros()
  
  var size_x : int32 = r_image.bounds.hi.x
  var size_y : int32 = r_image.bounds.hi.y
  c.printf("low bounds are: %d, %d\n", r_image.bounds.lo.x, r_image.bounds.lo.y)
  c.printf("hi bounds are %d, %d\n", size_x, size_y)
  
  --c.printf("checking values %d vs %d \n", r_image[{
  -- Note: value of y must change in correspondence with the image bounds here.
  
  --for r in r_image do
    ----c.printf("value at %d, %d is %d\n", r.x,r.y, r_image[r].value)
  --end

  for y = r_image.bounds.lo.y, size_y+1, step do
    
    var base =  r_image.bounds.lo.x
    regentlib.assert(base == 0, "base should always start with 0 in liftX")
    var base1 = base - step
    var base2 = base + step
    
    -- Add parallelism at this stage. These two loops are just doing
    -- reductions, especially after step = 2.
    
    var x : int32
    for x = step, size_x - step, step*2 do
      r_image[{base+x, y}].value -= (r_image[{base1+x, y}].value + r_image[{base2+x, y}].value)/2
      --if step == 2 then
        --c.printf("base+x = %d, base1+x = %d, base2+x = %d\n", base+x, base1+x, base2+x)
      --end
    end
    
    -- Edge condition.
    if (x < size_x) then
      r_image[{base+x, y}].value -= r_image[{base1+x, y}].value 
    end
    
    for x = step*2, size_x - step, step*2 do
      --if step == 2 then
        --c.printf("base+x = %d, base1+x = %d, base2+x = %d\n", base+x, base1+x, base2+x)
      --end
      r_image[{base+x, y}].value += (r_image[{base1+x, y}].value + r_image[{base2+x, y}].value)/4 
    end
    
    if (x < size_x) then
      r_image[{base+x, y}].value += r_image[{base1+x, y}].value / 2 
    end
    
  end

  var ts_end = c.legion_get_current_time_in_micros()
  --c.printf("Lift x took %.3f sec.\n", (ts_end - ts_start) * 1e-6)
  -- spurious dependencies.
  return 1
end

-- Vertical lifting. Slightly more complicated than Horizontal lifting.
--
task liftY(r_image    : region(ispace(int2d), Pixel), 
          step : int32)
where
  reads writes(r_image.value)
do 
  var ts_start = c.legion_get_current_time_in_micros()
  
  var y0 : int32 = r_image.bounds.lo.y
  var x0 : int32 = r_image.bounds.lo.x
  -- I think this should be always true?
  --assert (y0 == 0 and x0 == 0)

  var size_x : int32 = r_image.bounds.hi.x
  var size_y : int32 = r_image.bounds.hi.y

  for y = step, size_y, step*2 do
    var base :int2d = {x0, y0 + y}
    -- FIXME: Does this always stay within range?
    var c1base :int2d = {x0, y0 + y - step}
    var c2base :int2d = c1base
    
    if (y + step) < size_y then 
      c2base = {x0, y0 + y + step}
    end
      
    var x : int32

    -- Linear filtering
    -- If we did only a few columns at a time? 
    for x = 0, size_x, step do
      r_image[{base.x + x, base.y}].value -= (r_image[c1base + {x,0}].value + r_image[c2base + {x,0}].value) / 2
      --c.printf("base, x: %d, y: %d; c1base, x: %d, y: %d; c2base, x: %d, y: %d;\n", base.x+x,base.y,c1base.x+x,c1base.y,c2base.x+x,c2base.y)
    end 
  end
  
  -- Do we really need a separate loop for this?
  for y = step*2, size_y, step*2 do
    var base :int2d = {x0 , y0 + y}
    -- FIXME: Does this always stay within range?
    var g1base :int2d = {x0, y0 + y - step}
    var g2base :int2d = g1base 
    if (y + step) < size_y then 
      g2base = {x0, y0 + y + step}
    end
    
    for x = 0, size_x, step do
      r_image[base + {x, 0}].value += (r_image[g1base + {x,0}].value + r_image[g2base + {x,0}].value) / 4
      --c.printf("base, x: %d, y: %d; c1base, x: %d, y: %d; c2base, x: %d, y: %d;\n", base.x+x,base.y,g1base.x+x,g1base.y,g2base.x+x,g2base.y)
    end 
  end

  var ts_end = c.legion_get_current_time_in_micros()
  --c.printf("Lift y took %.3f sec.\n", (ts_end - ts_start) * 1e-6)
  -- spurious dependencies.
  return 1
end

task unliftX(r_image    : region(ispace(int2d), Pixel), 
          step : int32)
where
  reads writes(r_image.value)
do 
  var ts_start = c.legion_get_current_time_in_micros()
  
  var size_x : int32 = r_image.bounds.hi.x
  var size_y : int32 = r_image.bounds.hi.y

  for y = r_image.bounds.lo.y, size_y+1, step do
    
  --for y = r_image.bounds.lo.y + 0, size_y, step do
    -- just x values will be affected by these here.
    var base = r_image.bounds.lo.x
    var base1 = base - step
    var base2 = base + step
    
    var x : int32
    -- even loop?
    for x = step*2, size_x - step, step*2 do
      r_image[{base+x, y}].value -= (r_image[{base1+x, y}].value + r_image[{base2+x, y}].value)/4 
    end
    
    if (x < size_x) then
      r_image[{base+x, y}].value -= r_image[{base1+x, y}].value / 2 
    end
    
    for x = step, size_x - step, step*2 do
      r_image[{base+x, y}].value += (r_image[{base1+x, y}].value + r_image[{base2+x, y}].value)/2
    end
    
    -- Edge condition.
    if (x < size_x) then
      r_image[{base+x, y}].value += r_image[{base1+x, y}].value 
    end
    
  end

  var ts_end = c.legion_get_current_time_in_micros()
  --c.printf("UnLift x took %.3f sec.\n", (ts_end - ts_start) * 1e-6)
  -- spurious dependencies.
  return 1
end

task saveImage(r_image : region(ispace(int2d), Pixel),
                filename : rawstring)
where
  reads(r_image.value)
do
  png.write_png_file(filename,
                     __physical(r_image.value),
                     __fields(r_image.value),
                     r_image.bounds)
end

task unliftY(r_image    : region(ispace(int2d), Pixel), 
          step : int32)
where
  reads writes(r_image.value)
do 
  var ts_start = c.legion_get_current_time_in_micros()
  
  var y0 : int32 = r_image.bounds.lo.y
  var x0 : int32 = r_image.bounds.lo.x
  -- I think this should be always true?
  --assert (y0 == 0 && x0 == 0)

  var size_x : int32 = r_image.bounds.hi.x
  var size_y : int32 = r_image.bounds.hi.y
  
  for y = step*2, size_y, step*2 do
    -- exactly same as in lift step
    var base :int2d = {x0 , y0 + y}
    var g1base :int2d = {x0, y0 + y - step}
    var g2base :int2d = g1base 
    if (y + step) < size_y then 
      g2base = {x0, y0 + y + step}
    end

    for x = 0, size_x, step do
      r_image[base + {x, 0}].value -= ((r_image[g1base + {x,0}].value + r_image[g2base + {x,0}].value) / 4)
    end 
  end

  for y = step, size_y, step*2 do 
    var base :int2d = {x0, y0 + y}
    var c1base :int2d = {x0, y0 + y - step}
    var c2base :int2d = c1base
    
    if (y + step) < size_y then 
      c2base = {x0, y0 + y + step}
    end
      
    -- If we did only a few columns at a time? 
    for x = 0, size_x, step do
      r_image[{base.x + x, base.y}].value += (r_image[c1base + {x,0}].value + r_image[c2base + {x,0}].value) / 2
    end 
  end
  
  var ts_end = c.legion_get_current_time_in_micros()
  --c.printf("Lift y took %.3f sec.\n", (ts_end - ts_start) * 1e-6)
  -- spurious dependencies.
  return 1
end

-- Modify this to salvage it somehow. Maybe write out to a file or something.
task printOutImage(r_image: region(ispace(int2d), Pixel))
where
  reads(r_image.value)
do
  var size_y = r_image.bounds.hi.y  
  var size_x = r_image.bounds.hi.x
  
  for y = 0, size_y, 10 do
    c.printf("\n")
    for x = 0, size_x, 10 do
      c.printf("%d  ", r_image[{x,y}].value) 
    end
  end
  c.printf("finished printing this image\n")
end

task fill_image(r_image : region(ispace(int2d), Pixel),
                r_mini_image : region(ispace(int2d), Pixel))
where 
  reads (r_mini_image.value), writes (r_image.value)
do
  --regentlib.assert(r_image.bounds.lo == r_image.bounds.lo, 'assert failed')
  var low_bounds :int2d = r_image.bounds.lo
  --c.printf("low bounds, combined: %d, %d; image: %d, %d\n", low_bounds.x, low_bounds.y, r_mini_image.bounds.lo.x, r_mini_image.bounds.lo.y)
  --c.printf("combined image bounds are : %d, %d. r_image bounds are: %d, %d\n", r_image.bounds.hi.x, r_image.bounds.hi.y, r_mini_image.bounds.hi.x, r_mini_image.bounds.hi.y)
   
  for p in r_mini_image do
      var index : int2d = low_bounds + p
      r_image[index].value = r_mini_image[p].value
  end
end

task sequential_fill_image(r_image : region(ispace(int2d), Pixel),
                r_mini_image : region(ispace(int2d), Pixel),
                edge    : int32)
where 
  reads (r_mini_image), writes (r_image)
do

  for i = 0, edge, 1 do
    var new_y :int32 = i*r_mini_image.bounds.hi.y
    for j = 0, edge, 1 do
       var new_x :int32 = j*r_mini_image.bounds.hi.x 
       -- Now, we can fill up the image r_combined with r_image stuff
       for pixel in r_mini_image do
         var index : int2d = {new_x, new_y} + pixel 
         r_image[index].value = r_mini_image[pixel].value
       end 
    end
  end
end

task toplevel()
  var config : WaveletConfig
  config:initialize_from_command()

  -- FIX the config file to get this and stuff, but for now this is fine.
  config.num_parallelism = 1
  
  var edge : int32 = 4
  var size_image = png.get_image_size(config.filename_image)
  
  var size_combined_image = {edge*size_image.x, edge*size_image.y}

  var r_mini_image = region(ispace(int2d, size_image), Pixel) 
  initialize(r_mini_image, config.filename_image) 
  
  -- Now we have the combined image, and we want to fill it up with the same
  -- data as in the original r_image.
  
  var r_image = region(ispace(int2d, size_combined_image), Pixel) 
  sequential_fill_image(r_image, r_mini_image, edge)
  -- Let's do it in regent style. We will divide the combined image into
  -- color based squares with exact bounds (width and height from size_image)  
  --var coloring = c.legion_domain_point_coloring_create()   
  --fill it up horizontally and vertically separately.
 
  --var color_count :int1d = 0
  --for i = 0, edge, 1 do
    --var new_y :int32 = i*size_image.y
    --for j = 0, edge, 1 do
       --var new_x :int32 = j*size_image.x  
       --c.legion_domain_point_coloring_color_domain(coloring, color_count,
       --rect2d {{new_x, new_y}, {new_x+size_image.x-1, new_y + size_image.y-1}})
       --color_count += 1
    --end
  --end

  --var colors = ispace(int1d, edge*edge)
  --var p_combined_image = partition(disjoint, r_image, coloring, colors)
  --c.legion_domain_point_coloring_destroy(coloring)
  
  ---- Parallelized initialization.
  --for i = 0, edge*edge, 1 do
    --fill_image(p_combined_image[i], r_mini_image)
  --end
    
  saveImage(r_image, 'original_combined.png')
  -- Should we destroy a partition after we are done using it?
  var step : uint32 = 1
  
  -- Once we divide r_image into colors, can we still get size of each color
  -- partition? Also we want to divide it such that its a bunch of rows.
  -- Will we need the exact same partition while decompressing?
  
  var size_x : uint32 = r_image.bounds.hi.x
  var size_y : uint32 = r_image.bounds.hi.y
  
  --printOutImage(r_image)
  -- Want to launch the while loop on each region separately.
  
  var x_parallelism = config.num_parallelism
  var x_coloring = c.legion_domain_point_coloring_create()
  var chunk_height = size_y / x_parallelism

  ---- FIXME: Deal with the case when its not a perfect multiple and stuff.
  for i = 0, x_parallelism, 1 do
    var start_y = i*chunk_height
    var end_y = start_y + chunk_height - 1

    if i == x_parallelism-1 then
      end_y = size_y
    end
    c.legion_domain_point_coloring_color_domain(x_coloring, [int1d] (i),rect2d {{0, start_y}, {size_x,end_y}})
  end

  var x_colors = ispace(int1d, x_parallelism)
  var p_x_combined_image = partition(disjoint, r_image, x_coloring, x_colors)
  c.legion_domain_point_coloring_destroy(x_coloring)
  
  var y_parallelism = config.num_parallelism
  var y_coloring = c.legion_domain_point_coloring_create()
  var chunk_width = size_x / y_parallelism
  
  for i = 0, y_parallelism, 1 do
    var start_x = i*chunk_width
    var end_x = start_x + chunk_width - 1
    if i == y_parallelism-1 then
      end_x = size_x
    end
    c.legion_domain_point_coloring_color_domain(y_coloring, [int1d](i), rect2d{{start_x, 0}, {end_x,size_y}})
  end
  var y_colors = ispace(int1d, y_parallelism) 
  var p_y_combined_image = partition(disjoint, r_image, y_coloring, y_colors)
  c.legion_domain_point_coloring_destroy(y_coloring)
  
  var token = 0

  while (step < size_x or step < size_y) do
    if (step < size_x) then
      -- Can I do partition.colors? why not?
      for i = 0, x_parallelism, 1 do 
         liftX(p_x_combined_image[i], step)
      end
    end

    if (step < size_y) then
      --token += liftY(r_image, step)
     for i = 0, y_parallelism, 1 do
       liftY(p_y_combined_image[i], step)
      end
     end
    step = step*2
    wait_for(token)
  end
  
  saveImage(r_image, 'lifted_combined.png')

  -- Let's do the unlifting to check if it works 
  -- Gets step to the highest value so we can run it in reverse....
  while (step*2 < size_x or step*2 < size_y) do 
    step *= 2
  end

  while (step >=1) do
    if (step < size_y) then -- vertical lifting 
      --token += unliftY(r_image, step)
      for i = 0, y_parallelism, 1 do
        unliftY(p_y_combined_image[i], step)
      end
    end

    if (step < size_x) then
      --token += unliftX(r_image, step)
      for i = 0, x_parallelism, 1 do 
        unliftX(p_x_combined_image[i], step)
      end
    end
    step /= 2
    wait_for(token)
  end
  

  -- sanity check: what is happening with our massive image.
  saveImage(r_image, 'unlifted_combined.png')
end

regentlib.start(toplevel)
