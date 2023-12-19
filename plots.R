df1 = read.csv("D:\\Projects\\Python\\PycharmProjects\\tf28\\credo_stream\\results\\axes_similarityPCA.txt", header = FALSE)
df2 = read.csv("D:\\Projects\\Python\\PycharmProjects\\tf28\\credo_stream\\results\\axes_similarityBORDER_REPLICATE.txt", header = FALSE)
df3 = read.csv("D:\\Projects\\Python\\PycharmProjects\\tf28\\credo_stream\\results\\axes_similarityBORDER_REFLECT101.txt", header = FALSE)
df4 = read.csv("D:\\Projects\\Python\\PycharmProjects\\tf28\\credo_stream\\results\\axes_similarityBORDER_CONSTANT.txt", header = FALSE)

#lista = unlist(lista)
x = 1:length(df1$V1) / length(df1$V1) * 100
y = df1$V3
plot(x = x, y = y, xlab = "Percent of data used for training (batch size 10^4 images) [%]", ylab = "Coordinate frames weighted distance [rad]", col="red")
points(x = x, y = y, col='red')
y_smooth = smooth.spline(x, y, spar=0.5)
lines(x = x, y = y_smooth$y, col ="red")
title("Coordinate frames weighted distance between PCA and Incremental PCA")


y = df2$V3
points(x = x, y = y, col = 'blue')
y_smooth = smooth.spline(x, y, spar=0.5)
lines(x = x, y = y_smooth$y, col ="blue")

y = df3$V3
points(x = x, y = y, col = 'green')
y_smooth = smooth.spline(x, y, spar=0.5)
lines(x = x, y = y_smooth$y, col ="green")


y = df4$V3
points(x = x, y = y, col = 'magenta')
y_smooth = smooth.spline(x, y, spar=0.5)
lines(x = x, y = y_smooth$y, col ="magenta")


legend(65, 0.105, legend=c("None", "B. Replicate", "B. Reflect","B. Constant"),  
       fill = c("red","blue","green", "magenta"), title="Image alignment"
)

f = function(n)
{
  format(round(n, 3), nsmall = 3)
}

for (a in seq(1,nrow(df1),2))
{
  ss = paste(a, ' & ', f(df1$V3[a]), ' & ', f(df2$V3[a]), ' & ', f(df3$V3[a]),' & ' , f(df4$V3[a]), '\\\\ \n', sep="")
  cat(ss)
}

####################################################################

# Image to view size: 1000 x 800

read.as.list = function(ff)
{
  x <- scan(ff, what="", sep="\n")
  return(as.list(x))
}


make.statisctic = function(x1, x2)
{
  Jaccard.Similarity = length(intersect(x2, x1)) / length(unique(c(x1, x2)))
  Overlap.Coefficient = length(intersect(x2, x1)) / (min(length(x1), length(x2)))
  return (c(Jaccard.Similarity, Overlap.Coefficient))
}

all.files = list("pca.res 3.txt","pca.res 2.8.txt","pca.res 2.6.txt","pca.res 2.4.txt",
  "pca.resBORDER_CONSTANT 3.txt","pca.resBORDER_CONSTANT 2.8.txt","pca.resBORDER_CONSTANT 2.6.txt","pca.resBORDER_CONSTANT 2.4.txt",
  "pca.resBORDER_REPLICATE 3.txt","pca.resBORDER_REPLICATE 2.8.txt","pca.resBORDER_REPLICATE 2.6.txt","pca.resBORDER_REPLICATE 2.4.txt",
  "pca.resBORDER_REFLECT101 3.txt","pca.resBORDER_REFLECT101 2.8.txt","pca.resBORDER_REFLECT101 2.6.txt","pca.resBORDER_REFLECT101 2.4.txt")

xx = list()
for (ff in all.files)
{
  ff = paste("d:\\Projects\\Python\\PycharmProjects\\tf28\\credo_stream\\results\\", ff, sep = "")
  xx[[length(xx) + 1]] = read.as.list(ff)
}


ff = "d:\\Projects\\Python\\PycharmProjects\\tf28\\credo_stream\\results\\pca.resBORDER_REPLICATE 2.4.txt"
x1 = read.as.list(ff)
ff = "d:\\Projects\\Python\\PycharmProjects\\tf28\\credo_stream\\results\\pca.resBORDER_CONSTANT 2.4.txt" 
x2 = read.as.list(ff)
ff = "d:\\Projects\\Python\\PycharmProjects\\tf28\\credo_stream\\results\\pca.resBORDER_REFLECT101 2.4.txt" 
x3 = read.as.list(ff)
ff = "d:\\Projects\\Python\\PycharmProjects\\tf28\\credo_stream\\results\\pca.res 2.4.txt" 
x4 = read.as.list(ff)

make.statisctic(x2, x1)
make.statisctic(x3, x1)
make.statisctic(x3, x4)

method.1 = 1:length(all.files)
method.2 = 1:length(all.files)

Jaccard.Similarity.l = list()
Overlap.Coefficient.l = list()
x.l = list()
y.l = list()
for (a in 1:length(all.files))
{
  for (b in 1:length(all.files))
  {
    f1 = paste("d:\\Projects\\Python\\PycharmProjects\\tf28\\credo_stream\\results\\", all.files[[a]], sep = "")
    x1 = read.as.list(f1)
    f1 = paste("d:\\Projects\\Python\\PycharmProjects\\tf28\\credo_stream\\results\\", all.files[[b]], sep = "")
    x2 = read.as.list(f1)
    
    xx = make.statisctic(x1, x2)
    Jaccard.Similarity.l[[length(Jaccard.Similarity.l) + 1]] = xx[1]
    Overlap.Coefficient.l[[length(Overlap.Coefficient.l) + 1]] = xx[2]
    
    x.l[[length(x.l) + 1]] = a
    y.l[[length(x.l) + 1]] = b
  }
}


Jaccard.Similarity.df = data.frame(method.1 = unlist(x.l), method.2 = unlist(y.l), Y = unlist(Jaccard.Similarity.l))
Overlap.Coefficient.l.df = data.frame(method.1 = unlist(x.l), method.2 = unlist(y.l), Y = unlist(Overlap.Coefficient.l))


cols <-function(n) {
  colorRampPalette(rev(c("red4","red2","tomato2","orange","gold1","forestgreen","darkgreen","blue")))(8)
}


library(ggplot2)
p = ggplot(data =  Jaccard.Similarity.df, mapping = aes(x = factor(method.1), y = factor(method.2))) +
  geom_tile(aes(fill = Y), colour = "white") +
  geom_text(aes(label = sprintf("%1.2f", Y)), vjust = 1) +
  #scale_fill_gradient(low = "yellow", high = "red") +
  theme_bw()# + theme(legend.position = "none")
p + theme(legend.key.size = unit(1, 'cm')) + scale_fill_gradientn(name = 'Jaccard index', colours = cols(8), limits=c(0, 1)) + xlab("Algorithm") + ylab("Algorithm") + ggtitle('Comparison of detected anomaly sets for different algorithms') + theme(text = element_text(size = 17))


p = ggplot(data =  Overlap.Coefficient.l.df, mapping = aes(x = factor(method.1), y = factor(method.2))) +
  geom_tile(aes(fill = Y), colour = "white") +
  geom_text(aes(label = sprintf("%1.2f", Y)), vjust = 1) +
  #scale_fill_gradient(low = "yellow", high = "red") +
  theme_bw()# + theme(legend.position = "none")
p + theme(legend.key.size = unit(1, 'cm')) + scale_fill_gradientn(name = 'Overlap coefficient', colours = cols(8), limits=c(0, 1)) + xlab("Algorithm") + ylab("Algorithm") + ggtitle('Comparison of detected anomaly sets for different algorithms') + theme(text = element_text(size = 17))



#################################################################

all.files = list(
  "pca.resBORDER_REFLECT101 2.4.txt",
  "pca.resBORDER_REFLECT101 550000-560000 2.4.txt",
  "pca.resBORDER_REFLECT101 450000-460000 2.4.txt",
  "pca.resBORDER_REFLECT101 350000-360000 2.4.txt",
  "pca.resBORDER_REFLECT101 250000-260000 2.4.txt",
  "pca.resBORDER_REFLECT101 150000-160000 2.4.txt",
  "pca.resBORDER_REFLECT101 50000-60000 2.4.txt"
  )

Jaccard.Similarity.l = list()
Overlap.Coefficient.l = list()
x.l = list()
y.l = list()
for (a in 1:length(all.files))
{
  for (b in 1:length(all.files))
  {
    f1 = paste("d:\\Projects\\Python\\PycharmProjects\\tf28\\credo_stream\\results\\", all.files[[a]], sep = "")
    x1 = read.as.list(f1)
    f1 = paste("d:\\Projects\\Python\\PycharmProjects\\tf28\\credo_stream\\results\\", all.files[[b]], sep = "")
    x2 = read.as.list(f1)
    
    xx = make.statisctic(x1, x2)
    Jaccard.Similarity.l[[length(Jaccard.Similarity.l) + 1]] = xx[1]
    Overlap.Coefficient.l[[length(Overlap.Coefficient.l) + 1]] = xx[2]
    
    x.l[[length(x.l) + 1]] = a
    y.l[[length(x.l) + 1]] = b
  }
}

Jaccard.Similarity.df = data.frame(method.1 = unlist(x.l), method.2 = unlist(y.l), Y = unlist(Jaccard.Similarity.l))
Overlap.Coefficient.l.df = data.frame(method.1 = unlist(x.l), method.2 = unlist(y.l), Y = unlist(Overlap.Coefficient.l))


cols <-function(n) {
  colorRampPalette(rev(c("red4","red2","tomato2","orange","gold1","forestgreen","darkgreen","blue")))(8)
}


library(ggplot2)
p = ggplot(data =  Jaccard.Similarity.df, mapping = aes(x = factor(method.1), y = factor(method.2))) +
  geom_tile(aes(fill = Y), colour = "white") +
  geom_text(aes(label = sprintf("%1.2f", Y)), vjust = 1, size = 10) +
  #scale_fill_gradient(low = "yellow", high = "red") +
  theme_bw()# + theme(legend.position = "none")
p + theme(legend.key.size = unit(1, 'cm')) + scale_fill_gradientn(name = 'Jaccard index', colours = cols(8)) + xlab("Algorithm") + ylab("Algorithm") + ggtitle('Comparison of detected anomaly sets for PCA and Incremental PCA') + theme(text = element_text(size = 17))


p = ggplot(data =  Overlap.Coefficient.l.df, mapping = aes(x = factor(method.1), y = factor(method.2))) +
  geom_tile(aes(fill = Y), colour = "white") +
  geom_text(aes(label = sprintf("%1.2f", Y)), vjust = 1, size = 10) +
  #scale_fill_gradient(low = "yellow", high = "red") +
  theme_bw()# + theme(legend.position = "none")
p + theme(legend.key.size = unit(1, 'cm')) + scale_fill_gradientn(name = 'Overlap coefficient', colours = cols(8)) + xlab("Algorithm") + ylab("Algorithm") + ggtitle('Comparison of detected anomaly sets for PCA and Incremental PCA') + theme(text = element_text(size = 17))


